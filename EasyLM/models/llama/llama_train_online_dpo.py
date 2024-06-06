import pprint
import math
import time
from tqdm import tqdm, trange
import copy

import mlxu
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
import flax
import torch
import wandb

from ...data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    global_norm, get_float_dtype_by_name, set_random_seed,
    get_weight_decay_mask, make_shard_and_gather_fns,
    with_sharding_constraint
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForSequenceClassification, FlaxLLaMAForTokenRegression
)
from transformers import GenerationConfig

try:
    from jax_smi import initialise_tracking
    initialise_tracking()
except ImportError:
    print("jax_smi not installed, skipping SMI tracking")


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    mesh_dim='1,-1,1', # (dp, fsdp, mp). The dp dimension must be 1 for our code to work properly. dp x fsdp x mp must equal the number of devices.
    dtype='bf16',
    load_llama_config_policy='',
    load_llama_config_reward='',
    update_llama_config_policy='',
    update_llama_config_reward='',
    load_checkpoint_policy='',
    load_checkpoint_reward='',
    load_dataset_state='',
    log_freq=1,
    save_model_freq=0,
    save_milestone_freq=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    llama=LLaMAConfig.get_default_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,

    max_continuation_len=16,
    mini_batch_size=1, # must be a divisor of (batch_size x 2), and a multiple of (dp x fsdp) dimensions
    use_tpu=False,
    # relatively dynamic flags
    num_epochs=1,
    ppo_epochs=4,
    warmup_epochs=0.0,
    lr=5e-7,
    beta=0.1,
    reward_gain=1.0,
    reward_bias=0.0,
    # relatively static flags
    temperature=0.7,
    # debugging flags
    max_steps_per_epoch=0,
    generate_only=False,
)

def detach(x):
    return jax.lax.stop_gradient(x)


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

def ppo_rollout(
    policy_train_state,
    policy_model,
    rng, batch, tokenizer,
):
    rng_generator = JaxRNG(rng)
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'))) # dim 0 is sharded across dp and fsdp axes, dim 1 is not sharded

    prompt_input_ids, prompt_attn_mask = batch['prompt_input_ids'], batch['prompt_attn_mask']
    PL = prompt_input_ids.shape[1]

    # rollout from current policy
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=FLAGS.temperature,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=FLAGS.max_continuation_len,
        # forced_eos_token_id=eos_token_id,
    )
    outputs = policy_model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attn_mask,
        generation_config=generation_config,
        params=policy_train_state.params['params'],
        prng_key=rng_generator(),
    )
    input_ids = outputs.sequences # (B, L)

    # NOTE: This is a hack because generate() weirdly forces the last token to be 0 instead of 2
    last_token_index = jnp.argmax(jnp.cumsum(jnp.where(input_ids == tokenizer.pad_token_id, 0, 1), axis=1), axis=1) # (B)
    input_ids = jnp.concatenate([input_ids, input_ids[:, -1:]], axis=1) # (B, L+1)
    input_ids = input_ids.at[jnp.arange(input_ids.shape[0]), last_token_index + 1].set(tokenizer.eos_token_id)
    input_ids = input_ids[:, :-1] # (B, L)

    attn_mask = jnp.where(input_ids == tokenizer.pad_token_id, 0, 1) # (B, L)
    position_ids = jnp.clip(jnp.cumsum(attn_mask, axis=1) - 1, 0, None) # (B, L)
    cont_input_ids = input_ids[:, PL:] # (B, CL)
    cont_attn_mask = attn_mask[:, PL:] # (B, CL)
    cont_position_ids = position_ids[:, PL:] # (B, CL)

    batch['input_ids'] = detach(input_ids)
    batch['attn_mask'] = detach(attn_mask)
    batch['cont_input_ids'] = detach(cont_input_ids)
    batch['cont_attn_mask'] = detach(cont_attn_mask)
    batch['cont_position_ids'] = detach(cont_position_ids)

    batch = with_sharding_constraint(batch, PS())

    return rng_generator(), batch

def dpo_forward_backward(
    policy_params, reference_params, reward_params,
    policy_model, reference_model, reward_model,
    rng, batch, tokenizer,
):
    rng_generator = JaxRNG(rng)
    batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

    input_ids, attn_mask = batch['input_ids'], batch['attn_mask']
    prompt_input_ids = batch['prompt_input_ids']
    reward_prompt_input_ids, reward_prompt_attn_mask = batch['reward_prompt_input_ids'], batch['reward_prompt_attn_mask']
    cont_input_ids, cont_attn_mask, cont_position_ids = batch['cont_input_ids'], batch['cont_attn_mask'], batch['cont_position_ids']
    PL = prompt_input_ids.shape[1]

    # run reward model
    reward_input_ids = jnp.concatenate([reward_prompt_input_ids, cont_input_ids], axis=1) # (B, PL+CL)
    reward_attn_mask = jnp.concatenate([reward_prompt_attn_mask, cont_attn_mask], axis=1) # (B, PL+CL)
    reward = reward_model(reward_input_ids, reward_attn_mask, params=reward_params['params'], dropout_rng=rng_generator()).logits # (B)
    # If the last token is not EOS, then we set the reward to -10
    reward_position_ids = jnp.clip(jnp.cumsum(reward_attn_mask, axis=1) - 1, 0, None) # (B, PL+CL)
    reward_last_token_index = jnp.argmax(reward_position_ids, axis=1) # (B)
    reward_last_token_id = jnp.take_along_axis(reward_input_ids, reward_last_token_index[:, None], axis=-1).squeeze(-1) # (B)
    reward = jnp.where(reward_last_token_id == tokenizer.eos_token_id, reward, -10.0)
    reward = jax.lax.stop_gradient(reward)
    score = reward * FLAGS.reward_gain + FLAGS.reward_bias # (B)
    score = jax.lax.stop_gradient(score)

    # reshape into adjacent pairs
    score_pairs = score.reshape(score.shape[0] // 2, 2)
    # get chosen mask (True when idx 1 works)
    chosen_mask = score_pairs[:, 1] > score_pairs[:, 0]

    # run forward pass on policy
    cont_logits = policy_model(input_ids, attn_mask, params=policy_params, dropout_rng=rng_generator()).logits[:, PL-1:-1, :] # (B, CL, V)
    cont_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    # split our logps into pairs
    cont_logps_pairs = cont_logps.reshape(cont_logps.shape[0] // 2, 2, cont_logps.shape[1])

    # run forward pass on reference
    cont_ref_logits = reference_model(input_ids, attn_mask, params=reference_params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1, :] # (B, CL, V)
    cont_ref_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_ref_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    cont_ref_logps = jax.lax.stop_gradient(cont_ref_logps)
    # split our logps into pairs
    cont_ref_logps_pairs = cont_ref_logps.reshape(cont_ref_logps.shape[0] // 2, 2, cont_ref_logps.shape[1])

    # split our logps into chosen and rejected
    # since they are paired, is just a patter of which idx is chosen
    policy_chosen_logps = jnp.where(chosen_mask[:, None], cont_logps_pairs[:, 1], cont_logps_pairs[:, 0])
    policy_rejected_logps = jnp.where(chosen_mask[:, None], cont_logps_pairs[:, 0], cont_logps_pairs[:, 1])
    reference_chosen_logps = jnp.where(chosen_mask[:, None], cont_ref_logps_pairs[:, 1], cont_ref_logps_pairs[:, 0])
    reference_rejected_logps = jnp.where(chosen_mask[:, None], cont_ref_logps_pairs[:, 0], cont_ref_logps_pairs[:, 1])
    
    # dpo loss
    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
        beta=FLAGS.beta,
    )

    # get the scores for the pairs
    chosen_scores = jnp.where(chosen_mask, score_pairs[:, 1], score_pairs[:, 0])
    rejected_scores = jnp.where(chosen_mask, score_pairs[:, 0], score_pairs[:, 1])

    stats = {
        'odpo/losses': detach(jnp.mean(losses)),
        'odpo/chosen_logp_rewards': detach(jnp.mean(chosen_rewards)),
        'odpo/rejected_logp_rewards': detach(jnp.mean(rejected_rewards)),
        'odpo/margin': detach(jnp.mean(chosen_rewards - rejected_rewards)),
        'tokens/responses_len_mean': detach(jnp.mean(jnp.sum(cont_attn_mask, axis=1))),
        'odpo/chosen_scores': detach(jnp.mean(chosen_scores)),
        'odpo/rejected_scores': detach(jnp.mean(rejected_scores)),
    }
    return losses.mean(), stats


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

    print("Loading dataset...")
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer, padding_side='left', truncation_side='left')
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))
    wrapped_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.DataLoader) else dataset

    prompt_batch_size = wrapped_dataset.config.batch_size
    steps_per_epoch = len(wrapped_dataset) // prompt_batch_size
    steps_per_epoch = steps_per_epoch if FLAGS.max_steps_per_epoch == 0 else min(steps_per_epoch, FLAGS.max_steps_per_epoch)
    total_steps = FLAGS.num_epochs * steps_per_epoch
    completion_batch_size = prompt_batch_size * 2
    assert FLAGS.mini_batch_size % 2 == 0, "mini_batch_size must be a multiple of 2"
    assert completion_batch_size % (FLAGS.mini_batch_size * FLAGS.optimizer.accumulate_gradient_steps) == 0
    grad_updates_per_step = completion_batch_size // (FLAGS.mini_batch_size * FLAGS.optimizer.accumulate_gradient_steps) * FLAGS.ppo_epochs
    grad_updates_per_epoch = steps_per_epoch * grad_updates_per_step
    total_grad_updates = total_steps * grad_updates_per_step
    lr_warmup_grad_updates = math.ceil(FLAGS.warmup_epochs * grad_updates_per_epoch)
    seq_length = wrapped_dataset.seq_length + FLAGS.max_continuation_len
    print(f'len(wrapped_dataset)={len(wrapped_dataset)}')
    print(f'prompt_batch_size={prompt_batch_size}')
    print(f'completion_batch_size={completion_batch_size}')
    print(f'steps_per_epoch={steps_per_epoch}')
    print(f'total_steps={total_steps}')
    print(f'grad_updates_per_step={grad_updates_per_step}')
    print(f'grad_updates_per_epoch={grad_updates_per_epoch}')
    print(f'total_grad_updates={total_grad_updates}')
    print(f'lr_warmup_grad_updates={lr_warmup_grad_updates}')

    print("Building model...")
    if FLAGS.load_llama_config_policy != '':
        llama_config_policy = LLaMAConfig.load_config(FLAGS.load_llama_config_policy)
    else:
        llama_config_policy = LLaMAConfig(**FLAGS.llama)
    if FLAGS.update_llama_config_policy != '':
        llama_config_policy.update(dict(eval(FLAGS.update_llama_config_policy)))
    llama_config_policy.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if llama_config_policy.vocab_size < wrapped_dataset.vocab_size:
        llama_config_policy.update(dict(vocab_size=wrapped_dataset.vocab_size))

    if FLAGS.load_llama_config_reward != '':
        llama_config_reward = LLaMAConfig.load_config(FLAGS.load_llama_config_reward)
    else:
        llama_config_reward = LLaMAConfig(**FLAGS.llama)
    if FLAGS.update_llama_config_reward != '':
        llama_config_reward.update(dict(eval(FLAGS.update_llama_config_reward)))
    llama_config_reward.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if llama_config_reward.vocab_size < wrapped_dataset.vocab_size:
        llama_config_reward.update(dict(vocab_size=wrapped_dataset.vocab_size))

    policy_model = FlaxLLaMAForCausalLM(llama_config_policy, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reference_model = FlaxLLaMAForCausalLM(llama_config_policy, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reward_model = FlaxLLaMAForSequenceClassification(llama_config_reward, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)

    print("Building optimizer...")
    FLAGS.optimizer.adamw_optimizer.init_lr = 0.0
    FLAGS.optimizer.adamw_optimizer.lr = FLAGS.lr
    FLAGS.optimizer.adamw_optimizer.end_lr = FLAGS.lr
    FLAGS.optimizer.adamw_optimizer.lr_warmup_steps = lr_warmup_grad_updates
    policy_optimizer, policy_optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    print("Initializing training state and pjitting...")
    def init_fn_policy(rng):
        rng_generator = JaxRNG(rng)
        params = policy_model.module.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config_policy.rng_keys()),
        )
        return TrainState.create(params=params, tx=policy_optimizer, apply_fn=None)
    def create_trainstate_from_params_policy(params):
        return TrainState.create(params=params, tx=policy_optimizer, apply_fn=None)
    train_state_shapes_policy = jax.eval_shape(init_fn_policy, next_rng()) # .params = {'params': {'transformer', 'lm_head'}} => .params = {'transformer', 'lm_head'}
    train_state_partition_policy = match_partition_rules(LLaMAConfig.get_partition_rules(), train_state_shapes_policy)
    shard_fns_policy, gather_fns_policy = make_shard_and_gather_fns(train_state_partition_policy, train_state_shapes_policy)
    sharded_init_fn_policy = pjit(
        init_fn_policy,
        in_shardings=PS(),
        out_shardings=train_state_partition_policy,
    )
    sharded_create_trainstate_from_params_policy = pjit(
        create_trainstate_from_params_policy,
        in_shardings=(train_state_partition_policy.params, ),
        out_shardings=train_state_partition_policy,
        donate_argnums=(0, ),
    )

    def init_fn_reward(rng):
        rng_generator = JaxRNG(rng)
        params = reward_model.module.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config_reward.rng_keys()),
        )
        # I construct a train state, but later we will just use the params.
        return TrainState.create(params=params, tx=policy_optimizer, apply_fn=None)
    train_state_shapes_reward = jax.eval_shape(init_fn_reward, next_rng()) # .params = {'params': {'transformer', 'lm_head'}} => .params = {'transformer', 'lm_head'}
    # TODO: do the partition rules correctly apply when you only return the params?
    # something might break...
    train_state_partition_reward = match_partition_rules(LLaMAConfig.get_partition_rules(), train_state_shapes_reward)
    shard_fns_reward, _ = make_shard_and_gather_fns(train_state_partition_reward, train_state_shapes_reward)
    sharded_init_fn_reward = pjit(
        init_fn_reward,
        in_shardings=PS(),
        out_shardings=train_state_partition_reward,
    )

    def ppo_rollout_wrapper(
        policy_train_state,
        rng, batch,
    ):
        return ppo_rollout(
            policy_train_state,
            policy_model,
            rng, batch, tokenizer,
        )
    sharded_ppo_rollout = pjit(
        ppo_rollout_wrapper,
        in_shardings=(train_state_partition_policy, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),  # rng
    )
    def dpo_forward_backward_wrapper(
        policy_train_state, reference_params, reward_params,
        rng, batch,
    ):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        loss_and_metrics = lambda policy_params: dpo_forward_backward(
            policy_params['params'], reference_params, reward_params,
            policy_model, reference_model, reward_model,
            rng_generator(), batch, tokenizer,
        )
        grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
        (_, metrics), grads = grad_fn(policy_train_state.params)
        policy_train_state = policy_train_state.apply_gradients(grads=grads)
        return policy_train_state, rng_generator(), batch, metrics
    
    sharded_dpo_forward_backward = pjit(
        dpo_forward_backward_wrapper,
        in_shardings=(train_state_partition_policy, train_state_partition_policy.params, train_state_partition_reward.params, PS(), PS()),
        out_shardings=(train_state_partition_policy, PS(), PS(), PS()),
        donate_argnums=(0, 3),  # policy train state and rng
    )

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )
    def save_checkpoint(policy_train_state, step, milestone=False):
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config_policy.to_dict(),
        )
        checkpointer.save_all(
            train_state=policy_train_state,
            gather_fns=gather_fns_policy,
            metadata=metadata,
            milestone=milestone,
            step=step,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        # Load policy
        policy_train_state, policy_params = None, None
        if FLAGS.load_checkpoint_policy != '':
            print("Loading checkpoint (policy) ... (may take time to download)")
            policy_train_state, policy_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_policy, train_state_shapes_policy, shard_fns_policy)
            print("Checkpoint (policy) loaded.")
        if policy_train_state is None:
            if policy_params is None:
                policy_train_state = sharded_init_fn_policy(next_rng())
            else:
                if not FLAGS.use_tpu:
                    policy_params = flax.core.frozen_dict.unfreeze(policy_params)
                policy_train_state = sharded_create_trainstate_from_params_policy(policy_params)
                del policy_params

        # Load reference
        reference_params = None
        if FLAGS.load_checkpoint_policy != '':
            print("Loading checkpoint (reference) ... (may take time to download)")
            reference_train_state, reference_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_policy, train_state_shapes_policy, shard_fns_policy)
            print("Checkpoint (reference) loaded.")
            assert reference_train_state is None
        if reference_params is None:
            reference_params = copy.deepcopy(policy_train_state.params)
        else:
            if not FLAGS.use_tpu:
                reference_params = flax.core.frozen_dict.unfreeze(reference_params)

        # Load reward
        reward_params = None
        if FLAGS.load_checkpoint_reward != '':
            print("Loading checkpoint (reward) ... (may take time to download)")
            reward_train_state, reward_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_reward, train_state_shapes_reward, shard_fns_reward)
            print("Checkpoint (reward) loaded.")
            assert reward_train_state is None
        else:
            if reward_params is None:
                reward_params = sharded_init_fn_reward(next_rng()).params
            else:
                if not FLAGS.use_tpu:
                    reward_params = flax.core.frozen_dict.unfreeze(reward_params)

        sharded_rng = next_rng()

        global_step = 0
        for epoch in trange(0, FLAGS.num_epochs, ncols=0, position=0):
            for step, batch in zip(trange(0, steps_per_epoch, ncols=0, position=1), dataset):
                global_step += 1
                # jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                # Prepare batch for multi-rollout. The same prompts appear adjacently, so they end up in the same backward minibatch and there's some contrastive signal.
                batch = {k: jnp.repeat(v, 2, axis=0) for k, v in batch.items()}

                t0 = time.time()
                t = time.time()
                sharded_rng, batch = sharded_ppo_rollout(policy_train_state, sharded_rng, batch)
                batch['cont_position_ids'].block_until_ready()
                # If we do not use jax.device_get() to convert into numpy array first, we will get an error when iterating a sharded array with dim >= 100
                batch = {k: jax.device_get(v) for k, v in batch.items()}
                time_rollout = time.time() - t
                # jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                if FLAGS.generate_only:
                    stats = {
                        'time/dpo/rollout': time_rollout,
                    }
                    queries = tokenizer.batch_decode(batch['prompt_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    responses = tokenizer.batch_decode(batch['cont_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    rows = [[q, r] for q, r in zip(queries, responses)]
                    stats['game_log'] = wandb.Table(columns=['query', 'response'], rows=rows)
                    logger.log(stats)
                    continue

                t = time.time()
                assert batch['input_ids'].shape[0] % FLAGS.mini_batch_size == 0
                all_mbs, all_stats = [], []
                for ppo_epoch in range(FLAGS.ppo_epochs):
                    for mb_start in range(0, batch['input_ids'].shape[0], FLAGS.mini_batch_size):
                        mb_end = mb_start + FLAGS.mini_batch_size
                        mb_batch = {k: v[mb_start:mb_end] for k, v in batch.items()}
                        policy_train_state, sharded_rng, mb_batch, stats = sharded_dpo_forward_backward(
                            policy_train_state, reference_params, reward_params, sharded_rng, mb_batch
                        )
                        jax.tree_map(lambda x: x.block_until_ready(), policy_train_state.params)
                        jax.tree_map(lambda x: x.block_until_ready(), policy_train_state.opt_state)
                        mb_batch = {k: jax.device_get(v) for k, v in mb_batch.items()}
                        stats = {k: jax.device_get(v) for k, v in stats.items()}
                        all_mbs.append(mb_batch)
                        all_stats.append(stats)
                batch = {k: jnp.concatenate([mb[k] for mb in all_mbs], axis=0) for k in all_mbs[0].keys()}
                stats = {k: jnp.mean(jnp.stack([s[k] for s in all_stats], axis=0), axis=0) for k in all_stats[0].keys()}
                time_forward_backward = time.time() - t
                time_total = time.time() - t0
                # jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                print(f"step={global_step}, time_rollout={time_rollout:.2f}, time_forward_backward={time_forward_backward:.2f}, time_total={time_total:.2f}")

                if FLAGS.log_freq > 0 and global_step % FLAGS.log_freq == 0:
                    stats = {k: float(v) for k, v in stats.items()}
                    stats.update({
                        'odpo/learning_rate_policy': policy_optimizer_info['learning_rate_schedule'](global_step * grad_updates_per_step).item(),
                        'time/odpo/rollout': time_rollout,
                        'time/odpo/forward_backward': time_forward_backward,
                        'time/odpo/total': time_total,
                    })
                    queries = tokenizer.batch_decode(batch['prompt_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    responses = tokenizer.batch_decode(batch['cont_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    rows = [[q, r] for q, r in zip(queries, responses)]
                    stats['game_log'] = wandb.Table(columns=['query', 'response'], rows=rows)
                    logger.log(stats)

                if FLAGS.save_milestone_freq > 0 and global_step % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(policy_train_state, step=global_step, milestone=True)
                elif FLAGS.save_model_freq > 0 and global_step % FLAGS.save_model_freq == 0:
                    save_checkpoint(policy_train_state, step=global_step)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0 or FLAGS.save_milestone_freq > 0:
                save_checkpoint(policy_train_state, step=global_step, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
