gcloud alpha compute tpus tpu-vm ssh jiachengl-v3-512 --zone=us-east1-d --project=ai2-tpu --worker=all --command="\
export WANDB_API_KEY=$WANDB_API_KEY; \
cd n-tulu-ppo-jax; \
git pull; \
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; \
python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,32,16' \
    --load_llama_config_policy='70b' \
    --load_llama_config_reward='70b' \
    --load_checkpoint_policy='params::gs://hamishi-east1/easylm/llama2/tulu2_70b_fixed/tulu2_70b_fixed/f1f893b8fe3947f29f6a596773289178/streaming_params' \
    --load_checkpoint_reward='params::gs://hamishi-east1/rm/tulu_2_70b_base_rm_uf/05d1d06e22a748ccb4c82228f0162ef0/streaming_params_7612' \
    --tokenizer.vocab_file='gs://jiachengl-east1/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='tulu_prompt' \
    --train_dataset.tulu_prompt_dataset.path='gs://hamishi-east1/easylm/data/converted_pref_data/ultrafeedback_mean_aspects_cleaned.jsonl' \
    --train_dataset.tulu_prompt_dataset.seq_length=1024 \
    --max_continuation_len=1024 \
    --train_dataset.tulu_prompt_dataset.batch_size=32 \
    --rollouts_per_prompt=1 \
    --mini_batch_size=32 \
    --train_dataset.tulu_prompt_dataset.num_workers=16 \
    --train_dataset.tulu_prompt_dataset.remove_truncated_samples=True \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --warmup_epochs=0.0125 \
    --policy_freeze_epochs=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='train_v3.1.2_v3.1_70b_b32' \
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
