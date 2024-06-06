gcloud alpha compute tpus tpu-vm ssh jiachengl-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="\
export WANDB_API_KEY=$WANDB_API_KEY; \
cd n-tulu-ppo-jax; \
git pull; \
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; \
python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,64,4' \
    --load_llama_config_policy='13b' \
    --load_llama_config_reward='13b' \
    --load_checkpoint_policy='params::gs://hamishi-east1/easylm/llama2/tulu2_13b_fixed/tulu2_13b_fixed/455af914503740be9664497dae996762/streaming_params' \
    --load_checkpoint_reward='params::gs://hamishi-east1/rm/tulu2_13b_ultrafeedback_rm/7371c411dcfd4b09994aaa50a3a07128/streaming_params_1903' \
    --tokenizer.vocab_file='gs://jiachengl-east1/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='tulu_prompt' \
    --train_dataset.tulu_prompt_dataset.path='gs://hamishi-east1/easylm/data/converted_pref_data/ultrafeedback_mean_aspects_cleaned.jsonl' \
    --train_dataset.tulu_prompt_dataset.seq_length=1024 \
    --max_continuation_len=1024 \
    --train_dataset.tulu_prompt_dataset.batch_size=512 \
    --rollouts_per_prompt=1 \
    --mini_batch_size=64 \
    --train_dataset.tulu_prompt_dataset.num_workers=16 \
    --train_dataset.tulu_prompt_dataset.remove_truncated_samples=True \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --warmup_epochs=0.1 \
    --policy_freeze_epochs=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='train_v3.2_v3_interleave-fwd-bwd_nofreeze' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='/home/jiachengl/wandb' \
    --logger.output_dir='gs://jiachengl-east1/n-tulu-ppo-jax/' \
    --use_tpu=True \
    --ppo_epochs=1 \
    --lr=1e-6 \
    --kl_coef=0.05 \
    --reward_gain=1.0 --reward_bias=0.0 \
    --save_milestone_freq=10000 \
    --num_epochs=1 \
    --max_steps_per_epoch=0 \
    --generate_only=False \
    &> /home/jiachengl/all.log & \
"
