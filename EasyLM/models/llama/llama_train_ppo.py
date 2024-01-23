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
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule, FlaxLLaMAForCausalLM
)
from transformers import GenerationConfig, FlaxLogitsProcessorList


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    mesh_dim='1,-1,1',
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

    use_tpu=False,
    num_epochs=4,
    max_steps_per_epoch=0,
    max_continuation_len=16,
    ppo_epochs=4,
    mini_batch_size=1,
    temperature=0.7,
    kl_coef=0.2,
    whiten_rewards=False,
    gamma=1.0,
    lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    # max_grad_norm=1.0,
)


def whiten(rewards, mask, shift_mean=True):
    rewards = rewards * mask
    mean = jnp.sum(rewards, axis=-1, keepdims=True) / jnp.sum(mask, axis=-1, keepdims=True)
    rewards = rewards - mean
    if shift_mean:
        rewards = rewards + jnp.mean(rewards, axis=-1, keepdims=True)
    std = jnp.sqrt(jnp.sum(rewards ** 2, axis=-1, keepdims=True) / jnp.sum(mask, axis=-1, keepdims=True))
    rewards = rewards / std
    return rewards

def masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.sum(mask)

def detach(x):
    return jax.lax.stop_gradient(x)

# # TODO: verify this implementation (this was written by ChatGPT)
# # TODO: this takes too long to compile! why??
# def clip_grad(grad, max_grad_norm):
#     norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad)]))
#     clip = lambda x: jnp.where(norm < max_grad_norm, x, x * max_grad_norm / (norm + 1e-6))
#     return jax.tree_util.tree_map(clip, grad)


def ppo_loss(
    policy_model, value_model,
    policy_params, value_params,
    input_ids, attn_mask, cont_input_ids, cont_attn_mask, old_cont_logps, old_cont_values, advantages, returns,
):
    PL = input_ids.shape[1] - cont_input_ids.shape[1]

    # run forward pass on policy
    new_cont_logits = policy_model(input_ids, attn_mask, params=policy_params['params']).logits[:, PL-1:-1, :] # (B, CL, V)
    new_cont_logps = jnp.take_along_axis(jax.nn.log_softmax(new_cont_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)

    ratio = jnp.exp(new_cont_logps - old_cont_logps)
    pg_losses = -advantages * ratio # (B, CL)
    pg_losses2 = -advantages * jnp.clip(ratio, 1.0 - FLAGS.cliprange, 1.0 + FLAGS.cliprange) # (B, CL)
    pg_loss = masked_mean(jnp.maximum(pg_losses, pg_losses2), cont_attn_mask)

    # run forward pass on value
    new_cont_values = value_model(input_ids, attn_mask, params=value_params['params']).logits[:, PL-1:-1, 0] # (B, CL)

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
    advantages = jax.lax.stop_gradient(advantages)
    return advantages, returns

def ppo_step(
    policy_train_state, reference_train_state, value_train_state, reward_train_state,
    policy_model, reference_model, value_model, reward_model,
    batch, rng
):
    prompt_input_ids, prompt_attn_mask = batch['prompt_input_ids'], batch['prompt_attn_mask']
    reward_prompt_input_ids, reward_prompt_attn_mask = batch['reward_prompt_input_ids'], batch['reward_prompt_attn_mask']
    # weird results when sharding these, so only allow fsdp axis sharding.
    prompt_input_ids = with_sharding_constraint(prompt_input_ids, PS(('fsdp')))
    prompt_attn_mask = with_sharding_constraint(prompt_attn_mask, PS(('fsdp')))
    rng_generator = JaxRNG(rng)
    PL = prompt_input_ids.shape[1]

    timing = dict()
    t0 = time.time()

    # rollout from current policy
    t = time.time()
    pad_token_id = 0
    generation_config = GenerationConfig(
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=1,
        max_new_tokens=FLAGS.max_continuation_len,
    )
    outputs = policy_model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attn_mask,
        generation_config=generation_config,
        logits_processor=FlaxLogitsProcessorList(
            [FlaxTemperatureLogitsWarper(FLAGS.temperature)]
        ),
        prng_key=rng_generator(),
        params=policy_train_state.params['params'],
    )
    input_ids = outputs.sequences # (B, L)
    # shard things now since other parts of the pipeline should be fine.
    # input_ids = with_sharding_constraint(input_ids, PS(('dp', 'fsdp')))
    attn_mask = jnp.where(input_ids == pad_token_id, 0, 1) # (B, L)
    position_ids = jnp.clip(jnp.cumsum(attn_mask, axis=1) - 1, 0, None) # (B, L)
    cont_input_ids = input_ids[:, PL:] # (B, CL)
    cont_attn_mask = attn_mask[:, PL:] # (B, CL)
    cont_position_ids = position_ids[:, PL:] # (B, CL)
    timing['time/ppo/rollout'] = time.time() - t

    # # run reward model
    # t = time.time()
    # reward_input_ids = jnp.concatenate([reward_prompt_input_ids, cont_input_ids], axis=1) # (B, PL+CL)
    # reward_attn_mask = jnp.concatenate([reward_prompt_attn_mask, cont_attn_mask], axis=1) # (B, PL+CL)
    # reward_position_ids = jnp.clip(jnp.cumsum(reward_attn_mask, axis=1) - 1, 0, None) # (B, PL+CL)
    # reward_output = reward_model(reward_input_ids, reward_attn_mask, params=reward_train_state.params['params']).logits[:, :, 0] # (B, L)
    # last_token_index = jnp.argmax(reward_position_ids, axis=1) # (B)
    # scores = jnp.take_along_axis(reward_output, last_token_index[:, None], axis=-1).squeeze(-1) # (B)
    # scores = jax.lax.stop_gradient(scores)
    # timing['time/ppo/reward_forward_pass'] = time.time() - t

    # # run forward pass on policy
    # t = time.time()
    # cont_logits = policy_model(input_ids, attn_mask, params=policy_train_state.params['params']).logits[:, PL-1:-1, :] # (B, CL, V)
    # cont_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    # cont_logps = jax.lax.stop_gradient(cont_logps)
    # timing['time/ppo/policy_forward_pass'] = time.time() - t

    # # run forward pass on reference
    # t = time.time()
    # cont_ref_logits = reference_model(input_ids, attn_mask, params=reference_train_state.params['params']).logits[:, PL-1:-1, :] # (B, CL, V)
    # cont_ref_logps = jnp.take_along_axis(jax.nn.log_softmax(cont_ref_logits, axis=-1), cont_input_ids[:, :, None], axis=-1).squeeze(-1) # (B, CL)
    # cont_ref_logps = jax.lax.stop_gradient(cont_ref_logps)
    # timing['time/ppo/reference_forward_pass'] = time.time() - t

    # # run forward pass on value
    # t = time.time()
    # cont_values = value_model(input_ids, attn_mask, params=value_train_state.params['params']).logits[:, PL-1:-1, 0] # (B, CL)
    # cont_values = jax.lax.stop_gradient(cont_values)
    # timing['time/ppo/value_forward_pass'] = time.time() - t

    # # penalize rewards
    # t = time.time()
    # kl = cont_logps - cont_ref_logps # (B, CL)
    # non_score_reward = -FLAGS.kl_coef * kl # (B, CL)
    # cont_last_token_index = jnp.argmax(cont_position_ids, axis=1) # (B)
    # rewards = non_score_reward.at[:, cont_last_token_index].add(scores) # (B, CL)
    # rewards = jax.lax.stop_gradient(rewards)
    # timing['time/ppo/compute_rewards'] = time.time() - t

    # # compute advantages
    # t = time.time()
    # advantages, returns = compute_advantages(cont_values, rewards, cont_attn_mask) # (B, CL), (B, CL)
    # timing['time/ppo/compute_advantages'] = time.time() - t

    # t = time.time()
    # all_stats = []
    # for ppo_epoch in range(FLAGS.ppo_epochs):
    #     assert cont_input_ids.shape[0] % FLAGS.mini_batch_size == 0
    #     for mb_start in range(0, cont_input_ids.shape[0], FLAGS.mini_batch_size):
    #         mb_end = mb_start + FLAGS.mini_batch_size
    #         mb_input_ids = input_ids[mb_start:mb_end]
    #         mb_attn_mask = attn_mask[mb_start:mb_end]
    #         mb_cont_input_ids = cont_input_ids[mb_start:mb_end]
    #         mb_cont_attn_mask = cont_attn_mask[mb_start:mb_end]
    #         mb_cont_logps = cont_logps[mb_start:mb_end]
    #         mb_cont_values = cont_values[mb_start:mb_end]
    #         mb_advantages = advantages[mb_start:mb_end]
    #         mb_returns = returns[mb_start:mb_end]

    #         loss_fn = lambda policy_params, value_params: ppo_loss(
    #             policy_model, value_model,
    #             policy_params, value_params,
    #             mb_input_ids, mb_attn_mask, mb_cont_input_ids, mb_cont_attn_mask, mb_cont_logps, mb_cont_values, mb_advantages, mb_returns,
    #         )
    #         grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
    #         (_, stats), (policy_grads, value_grads) = grad_fn(policy_train_state.params, value_train_state.params)
    #         # policy_grads = clip_grad(policy_grads, max_grad_norm=FLAGS.max_grad_norm)
    #         # value_grads = clip_grad(value_grads, max_grad_norm=FLAGS.max_grad_norm)
    #         policy_train_state = policy_train_state.apply_gradients(grads=policy_grads)
    #         value_train_state = value_train_state.apply_gradients(grads=value_grads)
    #         all_stats.append(stats)
    # timing['time/ppo/optimize_step'] = time.time() - t

    # t = time.time()
    # stats = {k: jnp.mean(jnp.stack([s[k] for s in all_stats], axis=0), axis=0) for k in all_stats[0].keys()}
    # stats.update({
    #     'env/reward_mean': detach(jnp.mean(scores)),
    #     'objective/kl': detach(masked_mean(kl, cont_attn_mask)),
    #     'objective/kl_coef': FLAGS.kl_coef,
    #     'ppo/mean_non_score_reward': detach(masked_mean(non_score_reward, cont_attn_mask)),
    #     'ppo/mean_scores': detach(jnp.mean(scores)),
    #     'ppo/learning_rate': FLAGS.optimizer.adamw_optimizer.lr,
    # })
    # examples = {
    #     'prompt_input_ids': detach(prompt_input_ids),
    #     'cont_input_ids': detach(cont_input_ids),
    #     'scores': detach(scores),
    # }
    # timing['time/ppo/calc_stats'] = time.time() - t
    stats = {}
    examples = {
        'prompt_input_ids': detach(prompt_input_ids),
        'cont_input_ids': detach(cont_input_ids),
    }

    timing['time/ppo/total'] = time.time() - t0
    stats.update(timing)

    return policy_train_state, value_train_state, stats, examples


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
    # assert FLAGS.train_dataset.json_torch_dataset.batch_size % FLAGS.mini_batch_size == 0
    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer, padding_side='left', truncation_side='left')
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))
    wrapped_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.DataLoader) else dataset

    real_batch_size = wrapped_dataset.config.batch_size
    steps_per_epoch = len(wrapped_dataset) // real_batch_size
    steps_per_epoch = steps_per_epoch if FLAGS.max_steps_per_epoch == 0 else min(steps_per_epoch, FLAGS.max_steps_per_epoch)
    total_steps = FLAGS.num_epochs * steps_per_epoch
    seq_length = wrapped_dataset.seq_length + FLAGS.max_continuation_len
    print(f'len(wrapped_dataset)={len(wrapped_dataset)}')
    print(f'real_batch_size={real_batch_size}')
    print(f'steps_per_epoch={steps_per_epoch}')
    print(f'total_steps={total_steps}')

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
    value_model = FlaxLLaMAForCausalLM(llama_config_reward, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reference_model = FlaxLLaMAForCausalLM(llama_config_policy, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)
    reward_model = FlaxLLaMAForCausalLM(llama_config_reward, dtype=get_float_dtype_by_name(FLAGS.dtype), _do_init=False)

    print("Building optimizer...")
    if FLAGS.optimizer.adamw_optimizer.warmup_ratio > 0:
        FLAGS.optimizer.adamw_optimizer.lr_warmup_steps = math.ceil(FLAGS.optimizer.adamw_optimizer.warmup_ratio * total_steps)
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    print("Initializing training state and pjitting...")
    def init_fn_policy(rng):
        rng_generator = JaxRNG(rng)
        params = policy_model.module.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config_policy.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    def create_trainstate_from_params_policy(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
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
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
    def create_trainstate_from_params_reward(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)
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

    def train_step(
        policy_train_state, reference_train_state, value_train_state, reward_train_state,
        rng, batch,
    ):
        rng_generator = JaxRNG(rng)
        # batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        policy_train_state, value_train_state, stats, examples = ppo_step(
            policy_train_state, reference_train_state, value_train_state, reward_train_state,
            policy_model, reference_model, value_model, reward_model,
            batch, rng_generator()
        )
        # we dont return the ref train state because we dont want to update it
        return policy_train_state, value_train_state, rng_generator(), stats, examples
    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition_policy, train_state_partition_policy, train_state_partition_reward, train_state_partition_reward, PS(), PS()),
        out_shardings=(train_state_partition_policy, train_state_partition_reward, PS(), PS(), PS()),
        donate_argnums=(0, 2, 4),  # policy train state, value train state, and rng
    )

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )
    def save_checkpoint(policy_train_state, value_train_state, milestone=False):
        step = int(jax.device_get(policy_train_state.step))
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
        )
        # TODO: save value model

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        # start_step = int(jax.device_get(policy_train_state.step)) # TODO: fix this

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
        reference_train_state, reference_params = None, None
        if FLAGS.load_checkpoint_policy != '':
            print("Loading checkpoint (reference) ... (may take time to download)")
            reference_train_state, reference_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_policy, train_state_shapes_policy, shard_fns_policy)
            print("Checkpoint (reference) loaded.")
        if reference_train_state is None:
            if reference_params is None:
                reference_train_state = sharded_init_fn_policy(next_rng())
            else:
                if not FLAGS.use_tpu:
                    reference_params = flax.core.frozen_dict.unfreeze(reference_params)
                reference_train_state = sharded_create_trainstate_from_params_policy(reference_params)
                del reference_params

        # Load reward
        reward_train_state, reward_params = None, None
        if FLAGS.load_checkpoint_reward != '':
            print("Loading checkpoint (reward) ... (may take time to download)")
            reward_train_state, reward_params = checkpointer.load_trainstate_checkpoint(FLAGS.load_checkpoint_reward, train_state_shapes_reward, shard_fns_reward)
            print("Checkpoint (reward) loaded.")
        if reward_train_state is None:
            if reward_params is None:
                reward_train_state = sharded_init_fn_reward(next_rng())
            else:
                if not FLAGS.use_tpu:
                    reward_params = flax.core.frozen_dict.unfreeze(reward_params)
                reward_train_state = sharded_create_trainstate_from_params_reward(reward_params)
                del reward_params

        sharded_rng = next_rng()

        for epoch in trange(0, FLAGS.num_epochs, ncols=0, position=0):
            for step, batch in zip(trange(0, steps_per_epoch, ncols=0, position=1), dataset):
                policy_train_state, value_train_state, sharded_rng, stats, examples = sharded_train_step(
                    policy_train_state, reference_train_state, value_train_state, reward_train_state, sharded_rng, batch
                )

                if FLAGS.log_freq > 0 and step % FLAGS.log_freq == 0:
                    stats = {k: float(v) for k, v in stats.items()}
                    examples = jax.device_get(examples)
                    queries = tokenizer.batch_decode(examples['prompt_input_ids'], skip_special_tokens=True)
                    responses = tokenizer.batch_decode(examples['cont_input_ids'], skip_special_tokens=True)
                    # rewards = examples['scores']
                    # examples = [[q, r, float(reward)] for q, r, reward in zip(queries, responses, rewards)]
                    # stats['game_log'] = wandb.Table(columns=['query', 'response', 'reward'], rows=examples)
                    examples = [[q, r] for q, r in zip(queries, responses)]
                    stats['game_log'] = wandb.Table(columns=['query', 'response'], rows=examples)
                    logger.log(stats)
                    tqdm.write("\n" + pprint.pformat(stats) + "\n")

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(policy_train_state, value_train_state)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0:
                save_checkpoint(policy_train_state, value_train_state, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)
