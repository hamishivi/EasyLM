'''
Llama train script modified for DPO.
WIP!!!
'''
import pprint
import math
import time

from tqdm import tqdm, trange
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
import torch

from EasyLM.data import DatasetFactory, pad_out_to_full_batch
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    global_norm, get_float_dtype_by_name, set_random_seed,
    get_weight_decay_mask, make_shard_and_gather_fns,
    with_sharding_constraint
)
from EasyLM.models.olmo.olmo_model import (
    OLMoConfig, FlaxOLMoForCausalLMModule, OlmoTokenizer
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_olmo_config='',
    update_olmo_config='',
    load_checkpoint='',
    load_reference_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    num_epochs=0,
    tokenizer='',
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=OLMoConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    dpo_beta=0.1,
    dpo_label_smoothing=0.0,  # label smoothing for constrained DPO
    use_ipo=False,  # use IPO instead of DPO
    precalculate_reference_logps=False,  # precalculate reference logps, speeds up training and saves memory.
)


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             label_smoothing: float = 0.0,
             use_ipo: bool = False,):
    # modified from https://arxiv.org/pdf/2305.18290.pdf
    # also https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L375
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    # from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
    if use_ipo:
        # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        losses = (logits - 1/(2 * beta)) ** 2
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf;
        # label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing) - jax.nn.log_sigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * jax.lax.stop_gradient(policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * jax.lax.stop_gradient(policy_rejected_logps - reference_rejected_logps)

    return losses, chosen_rewards, rejected_rewards


def convert_logits_to_logps(logits, labels, loss_mask, label_pad_token_id):
    # shift labels and logits as usual (and loss mask)
    labels = labels[:, 1:]
    loss_mask = loss_mask[:, 1:]
    logits = logits[:, :-1]
    # set padding labels to 0
    labels = jnp.where(labels == label_pad_token_id, 0, labels)

    per_token_logps = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), axis=2, indices=labels[:,:,None]).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1)


def concatenated_forward(model, params, rng, batch, train=True):
    # concatenate chosen and rejected inputs
    # in tpu-land, we pad out to max length always so we can just concatenate naively :)
    len_chosen = batch['chosen_input_ids'].shape[0]
    concat_input_ids = jnp.concatenate([batch['chosen_input_ids'], batch['rejected_input_ids']], axis=0)
    concat_attn_mask = jnp.concatenate([batch['chosen_attn_mask'], batch['rejected_attn_mask']], axis=0)
    logits = model.apply(
        params, concat_input_ids, concat_attn_mask,
        deterministic=not train, rngs=rng,
    ).logits
    chosen_logits = logits[:len_chosen]
    rejected_logits = logits[len_chosen:]
    chosen_logps = convert_logits_to_logps(chosen_logits, batch['chosen_input_ids'], batch['chosen_loss_mask'], model.config.pad_token_id)
    rejected_logps = convert_logits_to_logps(rejected_logits, batch['rejected_input_ids'], batch['rejected_loss_mask'], model.config.pad_token_id)
    return chosen_logps, rejected_logps


def dpo_forward(
        model, params, rng, batch, beta, label_smoothing=0.0,
        use_ipo=False, reference_logps=None, reference_params=None
    ):
    assert reference_logps is not None or reference_params is not None, "Must provide either reference logps or reference params!"
    if reference_logps is not None:
        reference_chosen_logps, reference_rejected_logps = reference_logps
    else:
        reference_chosen_logps, reference_rejected_logps = concatenated_forward(model, reference_params, rng, batch)
    # jax doesnt have a 'no grad' manager, but if we stop gradients through the logps, this should work.
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, params, rng, batch)
    reference_chosen_logps = jax.lax.stop_gradient(reference_chosen_logps)
    reference_rejected_logps = jax.lax.stop_gradient(reference_rejected_logps)

    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
        use_ipo=use_ipo,
    )

    reward_accuracies = (chosen_rewards > rejected_rewards).astype(jnp.float32)
    metrics = {
        'loss': losses.mean(),
        'reward_accuracy': reward_accuracies.mean(),
        'score_chosen': chosen_rewards.mean(),
        'score_rejected': rejected_rewards.mean(),
        'score_margin': (chosen_rewards - rejected_rewards).mean(),
    }
    return losses.mean(), metrics


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = OlmoTokenizer.from_pretrained(FLAGS.tokenizer)
    # for olmo, we need to set the eos token to bos token
    tokenizer.bos_token = tokenizer.eos_token
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if isinstance(dataset, torch.utils.data.DataLoader):
        wrapped_dataset = dataset.dataset
    else:
        wrapped_dataset = dataset

    real_batch_size = wrapped_dataset.config.batch_size
    # for the scheduler, which only gets updated with 'real' grad steps
    simulated_batch_size = real_batch_size * FLAGS.optimizer.accumulate_gradient_steps
    steps_per_epoch = len(wrapped_dataset) // real_batch_size
    simulated_steps_per_epoch = len(wrapped_dataset) // simulated_batch_size
    print(f"Make sure your scheduler steps are based on the simulated batch size: {simulated_batch_size}!")
    print(f"Total simulated steps: {simulated_steps_per_epoch * FLAGS.num_epochs}")

    seq_length = wrapped_dataset.seq_length

    print("Building model...")
    if FLAGS.load_olmo_config != '':
        olmo_config = OLMoConfig.load_config(FLAGS.load_olmo_config)
    else:
        olmo_config = OLMoConfig(**FLAGS.olmo)

    if FLAGS.update_olmo_config != '':
        olmo_config.update(dict(eval(FLAGS.update_olmo_config)))

    olmo_config.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if olmo_config.vocab_size < wrapped_dataset.vocab_size:
        olmo_config.update(dict(vocab_size=wrapped_dataset.vocab_size))

    model = FlaxOLMoForCausalLMModule(
        olmo_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    print("Building optimizer...")
    if FLAGS.num_epochs > 0:
        total_simulated_steps = FLAGS.num_epochs * simulated_steps_per_epoch
        FLAGS.optimizer.adamw_optimizer.lr_decay_steps = total_simulated_steps
        if FLAGS.optimizer.adamw_optimizer.warmup_ratio > 0:
            FLAGS.optimizer.adamw_optimizer.lr_warmup_steps = math.ceil(FLAGS.optimizer.adamw_optimizer.warmup_ratio * total_simulated_steps)

    print(f"Total simulated steps: {total_simulated_steps}")
    print(f"Total simulated warmup steps: {FLAGS.optimizer.adamw_optimizer.lr_warmup_steps}")
    print(f"Total simulated decay steps: {FLAGS.optimizer.adamw_optimizer.lr_decay_steps}")

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(OLMoConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(olmo_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch, reference_logps=None, reference_train_state=None):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        loss_and_metrics = lambda params: dpo_forward(
            model,
            params,
            rng_generator(olmo_config.rng_keys()),
            batch,
            beta=FLAGS.dpo_beta,
            label_smoothing=FLAGS.dpo_label_smoothing,
            use_ipo=FLAGS.use_ipo,
            reference_logps=reference_logps,
            reference_params=reference_train_state.params if reference_train_state is not None else None,
        )
        grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        # additional metrics for tracking training
        metrics.update({
            "learning_rate": optimizer_info['learning_rate_schedule'](train_state.step // FLAGS.optimizer.accumulate_gradient_steps),
            "gradient_norm": global_norm(grads),
            "param_norm": global_norm(train_state.params),
        })
        # we dont return the ref train state because we dont want to update it
        return train_state, rng_generator(), metrics
    
    print("Initializing training state and pjitting...")
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        OLMoConfig.get_partition_rules(), train_state_shapes
    )
    params_partition = train_state_partition.params

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if not FLAGS.precalculate_reference_logps:
        in_shardings = (train_state_partition, PS(), PS(), PS(), train_state_partition)
    else:
        in_shardings = (train_state_partition, PS(), PS(), PS(), PS())
    sharded_train_step = pjit(
        train_step,
        in_shardings=in_shardings,
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),  # train state and rng
    )

    no_model_concate_forward = lambda train_state, rng, batch: concatenated_forward(model, train_state, rng, batch, train=False)
    sharded_concatenated_forward = pjit(
        no_model_concate_forward,
        in_shardings=(params_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            olmo_config=olmo_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            milestone=milestone,
        )

    mesh = OLMoConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params, reference_train_state, reference_params = None, None, None, None
        if FLAGS.load_checkpoint != '':
            print("Loading checkpoint... (may take time to download)")
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            print("Checkpoint loaded.")

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params
        
        if not FLAGS.precalculate_reference_logps:
            # reference params can be passed explicitly.
            if FLAGS.load_reference_checkpoint != '':
                reference_checkpoint = FLAGS.load_reference_checkpoint
            else:
                reference_checkpoint = FLAGS.load_checkpoint
            if FLAGS.load_checkpoint != '':
                print("Loading reference params... (may take time to download)")
                _, reference_params = checkpointer.load_trainstate_checkpoint(
                    reference_checkpoint, train_state_shapes, shard_fns
                )
                print("Reference params loaded.")
            else:
                print("Warning, your dpo reference params are not loaded from a checkpoint!")
                
            if reference_train_state is None and reference_params is None:
                # Initialize from scratch
                reference_train_state = sharded_init_fn(next_rng())
            elif reference_train_state is None and reference_params is not None:
                # Restore from params but initialize train_state
                reference_train_state = sharded_create_trainstate_from_params(reference_params)
                del reference_params
        else:
            # run an epoch to precalculate policy logps
            print("Precalculating policy logps...")
            all_reference_chosen_logps = []
            all_reference_rejected_logps = []
            # re-create a dataloader without shuffling so we can reference
            # using indices later.
            from torch.utils.data import DataLoader
            from transformers.data.data_collator import numpy_default_data_collator
            no_shuffle_dataset = DataLoader(
                dataset.dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                shuffle=False,
                collate_fn=numpy_default_data_collator,
            )
            for batch in tqdm(no_shuffle_dataset):
                batch.pop('indices')
                if batch['chosen_input_ids'].shape[0] < real_batch_size:
                    batch = pad_out_to_full_batch(real_batch_size, batch)
                rng = next_rng()
                rng_generator = JaxRNG(rng)
                # rng shouldnt matter here since i think?
                reference_chosen_logps, reference_rejected_logps = sharded_concatenated_forward(
                    train_state.params, rng_generator(olmo_config.rng_keys()), batch
                )
                all_reference_chosen_logps.append(jax.device_get(reference_chosen_logps))
                all_reference_rejected_logps.append(jax.device_get(reference_rejected_logps))
                del reference_chosen_logps, reference_rejected_logps
            all_reference_chosen_logps = jnp.concatenate(all_reference_chosen_logps, axis=0)
            all_reference_rejected_logps = jnp.concatenate(all_reference_rejected_logps, axis=0)                

        start_step = int(jax.device_get(train_state.step))
        start_epoch = start_step // steps_per_epoch
        start_step = start_step % steps_per_epoch

        sharded_rng = next_rng()

        if FLAGS.num_epochs > 0:
            epoch_counter = trange(start_epoch, FLAGS.num_epochs, ncols=0, position=0)
            step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
        else:
            epoch_counter = trange(start_epoch, math.ceil(FLAGS.total_steps / steps_per_epoch), ncols=0, position=0)
            step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        overall_step = 0
        for epoch in epoch_counter:
            for step, batch in zip(step_counter, dataset):
                start_time = time.time()
                if FLAGS.precalculate_reference_logps:
                    reference_train_state = None
                    # gather based on indices in batch
                    reference_chosen_logps = all_reference_chosen_logps[batch['indices']].squeeze(1)
                    reference_rejected_logps = all_reference_rejected_logps[batch['indices']].squeeze(1)
                    reference_logps = (reference_chosen_logps, reference_rejected_logps)
                else:
                    reference_logps = None

                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch, reference_logps, reference_train_state

                )
                step_time = time.time() - start_time
                overall_step += 1

                if step % FLAGS.log_freq == 0:
                    log_metrics = {
                        "train/step": overall_step,
                        "train/samples_seen": overall_step * real_batch_size,
                        "train/step_time": step_time,
                        "train/epoch": overall_step / steps_per_epoch,
                    }
                    log_metrics = jax.device_get(log_metrics)
                    log_metrics.update(metrics)
                    log_metrics = {k: float(v) for k, v in log_metrics.items()}
                    logger.log(log_metrics)
                    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(train_state)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0:
                save_checkpoint(train_state, milestone=True)
            # reset step counter
            if FLAGS.num_epochs > 0:
                step_counter = trange(0, steps_per_epoch, ncols=0, position=1)
            else:
                step_counter = trange(0, FLAGS.total_steps, ncols=0, position=1)

        # final log
        if FLAGS.log_freq > 0:
            log_metrics = {"step": step}
            metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics.update(metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        save_checkpoint(train_state, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
