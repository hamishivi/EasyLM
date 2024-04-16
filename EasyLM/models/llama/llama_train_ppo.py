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
    mini_batch_size=1, # must be a divisor of (batch_size x rollouts_per_prompt), and a multiple of (dp x fsdp) dimensions
    use_tpu=False,
    # relatively dynamic flags
    num_epochs=1,
    ppo_epochs=4,
    rollouts_per_prompt=1,
    warmup_epochs=0.0,
    policy_freeze_epochs=0.0,
    lr=1e-5,
    kl_coef=0.2,
    reward_gain=1.0,
    reward_bias=0.0,
    # relatively static flags
    temperature=0.7,
    whiten_rewards=False,
    gamma=1.0,
    lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    # debugging flags
    max_steps_per_epoch=0,
    generate_only=False,
)


def masked_sum(x, mask, axis=None):
    if axis is None:
        return jnp.sum(x * mask)
    else:
        return jnp.sum(x * mask, axis=axis)

def masked_mean(x, mask, axis=None):
    if axis is None:
        return jnp.sum(x * mask) / jnp.sum(mask)
    else:
        return jnp.sum(x * mask, axis=axis) / jnp.sum(mask, axis=axis)

def masked_var(x, mask, unbiased=True):
    mean = masked_mean(x, mask)
    centered_values = x - mean
    variance = masked_mean(jnp.square(centered_values), mask)
    if unbiased:
        mask_sum = jnp.sum(mask)
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def whiten(x, mask, shift_mean=True):
    mean, var = masked_mean(x, mask), masked_var(x, mask)
    whitened = (x - mean) / jnp.sqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

def detach(x):
    return jax.lax.stop_gradient(x)


def ppo_loss(
    policy_model, value_model,
    policy_params, value_params,
    rng,
    input_ids, attn_mask, cont_input_ids, cont_attn_mask, old_cont_logps, old_cont_values, advantages, returns,
):
    rng_generator = JaxRNG(rng)

    PL = input_ids.shape[1] - cont_input_ids.shape[1]

    # run forward pass on policy
    new_cont_logits = policy_model(input_ids, attn_mask, params=policy_params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1, :] # (B, CL, V)
    new_cont_logps = jnp.take_along_axis(jax.nn.log_softmax(new_cont_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)

    ratio = jnp.exp(new_cont_logps - old_cont_logps)
    pg_losses = -advantages * ratio # (B, CL)
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - FLAGS.cliprange, 1.0 + FLAGS.cliprange) # (B, CL)
    pg_loss = masked_mean(jnp.maximum(pg_losses, pg_losses2), cont_attn_mask)

    # run forward pass on value
    new_cont_values = value_model(input_ids, attn_mask, params=value_params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1] # (B, CL)

    new_cont_values_clipped = old_cont_values + jnp.clip(new_cont_values - old_cont_values, -FLAGS.cliprange_value, FLAGS.cliprange_value)
    vf_losses1 = jnp.square(new_cont_values - returns) # (B, CL)
    vf_losses2 = jnp.square(new_cont_values_clipped - returns) # (B, CL)
    vf_loss = 0.5 * masked_mean(jnp.maximum(vf_losses1, vf_losses2), cont_attn_mask)

    loss = pg_loss + FLAGS.vf_coef * vf_loss

    stats = {
        'ppo/loss/policy': detach(pg_loss),
        'ppo/loss/value': detach(vf_loss),
        'ppo/loss/total': detach(loss),
        'ppo/policy/ratios_mean': detach(masked_mean(ratio, cont_attn_mask)),
        'ppo/policy/advantages_mean': detach(masked_mean(advantages, cont_attn_mask)),
        'ppo/returns/mean': detach(masked_mean(returns, cont_attn_mask)),
        'ppo/val/vpred': detach(masked_mean(new_cont_values, cont_attn_mask)),
        'ppo/val/error': detach(masked_mean(jnp.square(new_cont_values - returns), cont_attn_mask)),
        'ppo/val/mean': detach(masked_mean(old_cont_values, cont_attn_mask)),
    }
    return loss, stats

def compute_advantages(values, rewards, mask):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = mask.shape[1]
    values = values * mask
    rewards = rewards * mask
    if FLAGS.whiten_rewards:
        rewards = whiten(rewards, mask, shift_mean=False)
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else jnp.zeros_like(values[:, t])
        delta = rewards[:, t] + FLAGS.gamma * nextvalues - values[:, t]
        lastgaelam = delta + FLAGS.gamma * FLAGS.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = jnp.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    advantages = whiten(advantages, mask, shift_mean=True)
    return advantages, returns

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

def ppo_forward_backward(
    policy_train_state, reference_params, value_train_state, reward_params,
    policy_model, reference_model, value_model, reward_model,
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

    # run forward pass on policy
    cont_logits = policy_model(input_ids, attn_mask, params=policy_train_state.params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1, :] # (B, CL, V)
    cont_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    cont_logps = jax.lax.stop_gradient(cont_logps)

    # run forward pass on reference
    cont_ref_logits = reference_model(input_ids, attn_mask, params=reference_params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1, :] # (B, CL, V)
    cont_ref_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_ref_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    cont_ref_logps = jax.lax.stop_gradient(cont_ref_logps)

    # run forward pass on value
    cont_values = value_model(input_ids, attn_mask, params=value_train_state.params['params'], dropout_rng=rng_generator()).logits[:, PL-1:-1] # (B, CL)
    cont_values = jax.lax.stop_gradient(cont_values)

    # penalize rewards
    kl = cont_logps - cont_ref_logps # (B, CL)
    kl = jax.lax.stop_gradient(kl)
    non_score_rewards = -FLAGS.kl_coef * kl # (B, CL)
    non_score_rewards = jax.lax.stop_gradient(non_score_rewards)
    cont_last_token_index = jnp.argmax(cont_position_ids, axis=1) # (B)
    rewards = non_score_rewards.at[jnp.arange(input_ids.shape[0]), cont_last_token_index].add(score) # (B, CL)
    rewards = jax.lax.stop_gradient(rewards)

    # compute advantages
    advantages, returns = compute_advantages(cont_values, rewards, cont_attn_mask) # (B, CL), (B, CL)
    advantages = jax.lax.stop_gradient(advantages)
    returns = jax.lax.stop_gradient(returns)

    batch.update({
        'reward': detach(reward),
    })
    stats = {
        'env/reward_mean': detach(jnp.mean(reward)),
        'objective/kl': detach(jnp.mean(masked_sum(kl, cont_attn_mask, axis=1))),
        'objective/kl_per_token': detach(masked_mean(kl, cont_attn_mask)),
        'objective/kl_coef': FLAGS.kl_coef,
        'ppo/mean_score_total': detach(jnp.mean(masked_sum(rewards, cont_attn_mask, axis=1))),
        'ppo/mean_non_score_reward': detach(masked_mean(non_score_rewards, cont_attn_mask)),
        'ppo/mean_non_score_reward_sum': detach(jnp.mean(masked_sum(non_score_rewards, cont_attn_mask, axis=1))),
        'ppo/mean_scores': detach(jnp.mean(score)),
        'ppo/std_scores': detach(jnp.std(score)),
        'tokens/responses_len_mean': detach(jnp.mean(jnp.sum(cont_attn_mask, axis=1))),
    }

    loss_fn = lambda policy_params, value_params: ppo_loss(
        policy_model, value_model,
        policy_params, value_params,
        rng_generator(),
        input_ids, attn_mask, cont_input_ids, cont_attn_mask, cont_logps, cont_values, advantages, returns,
    )
    grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
    (_, new_stats), (policy_grads, value_grads) = grad_fn(policy_train_state.params, value_train_state.params)
    policy_train_state = policy_train_state.apply_gradients(grads=policy_grads)
    value_train_state = value_train_state.apply_gradients(grads=value_grads)

    stats.update(new_stats)
    batch = with_sharding_constraint(batch, PS())

    return policy_train_state, value_train_state, rng_generator(), batch, stats


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
    completion_batch_size = prompt_batch_size * FLAGS.rollouts_per_prompt
    assert completion_batch_size % (FLAGS.mini_batch_size * FLAGS.optimizer.accumulate_gradient_steps) == 0
    grad_updates_per_step = completion_batch_size // (FLAGS.mini_batch_size * FLAGS.optimizer.accumulate_gradient_steps) * FLAGS.ppo_epochs
    grad_updates_per_epoch = steps_per_epoch * grad_updates_per_step
    total_grad_updates = total_steps * grad_updates_per_step
    lr_warmup_grad_updates = math.ceil(FLAGS.warmup_epochs * grad_updates_per_epoch)
    policy_freeze_grad_updates = math.ceil(FLAGS.policy_freeze_epochs * grad_updates_per_epoch)
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
    print(f'policy_freeze_grad_updates={policy_freeze_grad_updates}')

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

    policy_model = FlaxLLaMAForCausalLM(llama_config_policy, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False, mesh=mesh)
    value_model = FlaxLLaMAForTokenRegression(llama_config_reward, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False, mesh=mesh)
    reference_model = FlaxLLaMAForCausalLM(llama_config_policy, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False, mesh=mesh)
    reward_model = FlaxLLaMAForSequenceClassification(llama_config_reward, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False, mesh=mesh)

    print("Building optimizer...")
    FLAGS.optimizer.adamw_optimizer.init_lr = 0.0
    FLAGS.optimizer.adamw_optimizer.lr = FLAGS.lr
    FLAGS.optimizer.adamw_optimizer.end_lr = FLAGS.lr
    FLAGS.optimizer.adamw_optimizer.lr_warmup_steps = lr_warmup_grad_updates
    value_optimizer, value_optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)
    FLAGS.optimizer.adamw_optimizer.lr_freeze_steps = policy_freeze_grad_updates
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
        return TrainState.create(params=params, tx=value_optimizer, apply_fn=None)
    def create_trainstate_from_params_reward(params):
        return TrainState.create(params=params, tx=value_optimizer, apply_fn=None)
    train_state_shapes_reward = jax.eval_shape(init_fn_reward, next_rng()) # .params = {'params': {'transformer', 'lm_head'}} => .params = {'transformer', 'lm_head'}
    train_state_partition_reward = match_partition_rules(LLaMAConfig.get_partition_rules(), train_state_shapes_reward)
    shard_fns_reward, gather_fns_reward = make_shard_and_gather_fns(train_state_partition_reward, train_state_shapes_reward)
    sharded_init_fn_reward = pjit(
        init_fn_reward,
        in_shardings=PS(),
        out_shardings=train_state_partition_reward,
    )
    sharded_create_trainstate_from_params_reward = pjit(
        create_trainstate_from_params_reward,
        in_shardings=(train_state_partition_reward.params, ),
        out_shardings=train_state_partition_reward,
        donate_argnums=(0, ),
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
    def ppo_forward_backward_wrapper(
        policy_train_state, reference_params, value_train_state, reward_params,
        rng, batch,
    ):
        return ppo_forward_backward(
            policy_train_state, reference_params, value_train_state, reward_params,
            policy_model, reference_model, value_model, reward_model,
            rng, batch, tokenizer,
        )
    sharded_ppo_forward_backward = pjit(
        ppo_forward_backward_wrapper,
        in_shardings=(train_state_partition_policy, train_state_partition_policy.params, train_state_partition_reward, train_state_partition_reward.params, PS(), PS()),
        out_shardings=(train_state_partition_policy, train_state_partition_reward, PS(), PS(), PS()),
        donate_argnums=(0, 2, 4),  # policy train state, value train state, and rng
    )

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )
    def save_checkpoint(policy_train_state, value_train_state, step, milestone=False):
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
        checkpointer.save_all(
            train_state=value_train_state,
            gather_fns=gather_fns_reward,
            metadata=metadata,
            milestone=milestone,
            step=step,
            is_value=True,
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

        # Load value
        value_train_state, value_params = None, None
        if FLAGS.load_checkpoint_reward != '':
            print("Loading checkpoint (value) ... (may take time to download)")
            value_train_state, value_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_reward, train_state_shapes_reward, shard_fns_reward)
            print("Checkpoint (value) loaded.")
        if value_train_state is None:
            if value_params is None:
                value_train_state = sharded_init_fn_reward(next_rng())
            else:
                if not FLAGS.use_tpu:
                    value_params = flax.core.frozen_dict.unfreeze(value_params)
                value_train_state = sharded_create_trainstate_from_params_reward(value_params)
                del value_params

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
        if reward_params is None:
            reward_params = copy.deepcopy(value_train_state.params)
        else:
            if not FLAGS.use_tpu:
                reward_params = flax.core.frozen_dict.unfreeze(reward_params)

        sharded_rng = next_rng()

        global_step = 0
        for epoch in trange(0, FLAGS.num_epochs, ncols=0, position=0):
            for step, batch in zip(trange(0, steps_per_epoch, ncols=0, position=1), dataset):
                global_step += 1
                jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                # Prepare batch for multi-rollout. The same prompts appear adjacently, so they end up in the same backward minibatch and there's some contrastive signal.
                batch = {k: jnp.repeat(v, FLAGS.rollouts_per_prompt, axis=0) for k, v in batch.items()}

                t0 = time.time()
                t = time.time()
                sharded_rng, batch = sharded_ppo_rollout(policy_train_state, sharded_rng, batch)
                batch['cont_position_ids'].block_until_ready()
                # If we do not use jax.device_get() to convert into numpy array first, we will get an error when iterating a sharded array with dim >= 100
                batch = {k: jax.device_get(v) for k, v in batch.items()}
                time_rollout = time.time() - t
                jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                if FLAGS.generate_only:
                    stats = {
                        'time/ppo/rollout': time_rollout,
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
                        policy_train_state, value_train_state, sharded_rng, mb_batch, stats = sharded_ppo_forward_backward(
                            policy_train_state, reference_params, value_train_state, reward_params, sharded_rng, mb_batch
                        )
                        jax.tree_map(lambda x: x.block_until_ready(), policy_train_state.params)
                        jax.tree_map(lambda x: x.block_until_ready(), policy_train_state.opt_state)
                        jax.tree_map(lambda x: x.block_until_ready(), value_train_state.params)
                        jax.tree_map(lambda x: x.block_until_ready(), value_train_state.opt_state)
                        mb_batch = {k: jax.device_get(v) for k, v in mb_batch.items()}
                        stats = {k: jax.device_get(v) for k, v in stats.items()}
                        all_mbs.append(mb_batch)
                        all_stats.append(stats)
                batch = {k: jnp.concatenate([mb[k] for mb in all_mbs], axis=0) for k in all_mbs[0].keys()}
                stats = {k: jnp.mean(jnp.stack([s[k] for s in all_stats], axis=0), axis=0) for k in all_stats[0].keys()}
                time_forward_backward = time.time() - t
                time_total = time.time() - t0
                jax.profiler.save_device_memory_profile('/dev/shm/memory.prof')

                print(f"step={global_step}, time_rollout={time_rollout:.2f}, time_forward_backward={time_forward_backward:.2f}, time_total={time_total:.2f}")

                if FLAGS.log_freq > 0 and global_step % FLAGS.log_freq == 0:
                    stats = {k: float(v) for k, v in stats.items()}
                    stats.update({
                        'ppo/learning_rate_policy': policy_optimizer_info['learning_rate_schedule'](global_step * grad_updates_per_step).item(),
                        'ppo/learning_rate_value': value_optimizer_info['learning_rate_schedule'](global_step * grad_updates_per_step).item(),
                        'time/ppo/rollout': time_rollout,
                        'time/ppo/forward_backward': time_forward_backward,
                        'time/ppo/total': time_total,
                    })
                    queries = tokenizer.batch_decode(batch['prompt_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    responses = tokenizer.batch_decode(batch['cont_input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    rewards = batch['reward']
                    rows = [[q, r, float(reward)] for q, r, reward in zip(queries, responses, rewards)]
                    stats['game_log'] = wandb.Table(columns=['query', 'response', 'reward'], rows=rows)
                    logger.log(stats)

                if FLAGS.save_milestone_freq > 0 and global_step % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state, step=global_step, milestone=True)
                elif FLAGS.save_model_freq > 0 and global_step % FLAGS.save_model_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state, step=global_step)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0 or FLAGS.save_milestone_freq > 0:
                save_checkpoint(policy_train_state, value_train_state, step=global_step, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
