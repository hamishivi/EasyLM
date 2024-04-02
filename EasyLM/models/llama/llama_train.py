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

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint, average_metrics
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    num_epochs=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


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
    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
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

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, tokenizer
        )

    seq_length = wrapped_dataset.seq_length

    print("Building model...")
    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < wrapped_dataset.vocab_size:
        llama_config.update(dict(vocab_size=wrapped_dataset.vocab_size))

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype), mesh=mesh
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
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((8, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((8, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((8, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params, batch['input_tokens'], batch['attention_mask'],
            deterministic=True, rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        def loss_and_accuracy(params):
            logits = model.apply(
                params, batch['input_tokens'], batch['attention_mask'],
                deterministic=False, rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step // FLAGS.optimizer.accumulate_gradient_steps),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    print("Initializing training state and pjitting...")
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

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

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            milestone=milestone,
        )

    
    with mesh:
        train_state, restored_params = None, None
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

        start_step = int(jax.device_get(train_state.step))
        start_epoch = start_step // steps_per_epoch
        start_step = start_step % steps_per_epoch

        # if FLAGS.save_model_freq > 0:
        #     print("Initial save...")
        #     save_checkpoint(train_state)

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
                if isinstance(batch, (list, tuple)):
                    batch = {
                        'input_tokens': batch['input_tokens'],
                        'attention_mask': batch['attention_mask'],
                        'loss_masks': batch['loss_masks'],
                        'target_tokens': batch['target_tokens'],
                    }
                # just measuring the train step time.
                start_time = time.time()
                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )
                step_time = time.time() - start_time
                overall_step += 1

                if step % FLAGS.log_freq == 0:
                    if FLAGS.eval_steps > 0:
                        eval_metric_list = []
                        for batch in eval_dataset:
                            if isinstance(batch, (list, tuple)):
                                batch = {
                                    'tokens': batch[0],
                                    'loss_masks': batch[1],
                                }
                            sharded_rng, eval_metrics = sharded_eval_step(
                                train_state, sharded_rng, batch
                            )
                            eval_metric_list.append(eval_metrics)
                        metrics.update(average_metrics(eval_metric_list))
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
                step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
            else:
                step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        # final log
        if FLAGS.log_freq > 0:
            log_metrics = {"step": step}
            metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics.update(metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        if True:#FLAGS.save_model_freq > 0:
            save_checkpoint(train_state, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
