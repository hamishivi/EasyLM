gcloud alpha compute tpus tpu-vm ssh jiachengl-tpu-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="cd n-tulu-ppo-jax; git pull; export WANDB_API_KEY='a46519994b4614615d5ce4aa8742ef19685a7cae'; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,32,8' \
    --load_llama_config_policy='7b' \
    --load_llama_config_reward='13b' \
    --load_checkpoint_policy='params::gs://hamishi-dev/easylm/llama2/tulu2_7b_fixed/263f4f758b194729b206d5adad2b50d7/streaming_params' \
    --load_checkpoint_reward='params::gs://jiachengl/ckpt/UltraRM-13b/streaming_params' \
    --tokenizer.vocab_file='gs://hamishi-dev/easylm/llama/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='hf_prompt' \
    --train_dataset.text_processor.fields='[instruction]' \
    --train_dataset.hf_prompt_dataset.seq_length=1024 \
    --train_dataset.hf_prompt_dataset.batch_size=32 \
    --train_dataset.hf_prompt_dataset.num_workers=32 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.init_lr=1e-6 \
    --optimizer.adamw_optimizer.lr=1e-6 \
    --optimizer.adamw_optimizer.end_lr=1e-6 \
    --optimizer.adamw_optimizer.warmup_ratio=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='train_v1.5_v1.4_rg0.1_t1-32-8_b32_mb32' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='wandb' \
    --logger.output_dir='/home/jiachengl/runs/' \
    --use_tpu=True \
    --mini_batch_size=32 \
    --max_continuation_len=1024 \
    --max_steps_per_epoch=0 \
    --save_model_freq=0 \
    --reward_gain=0.1 --reward_bias=0.0 \
    --kl_coef=0.01 \
    --ppo_epochs=1 \
    &> ~/all.log &"
