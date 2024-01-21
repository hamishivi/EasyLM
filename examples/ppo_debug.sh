python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='-1,1,1' \
    --load_llama_config_policy='debug' \
    --load_llama_config_reward='debug' \
    --load_checkpoint_policy='' \
    --load_checkpoint_reward='' \
    --tokenizer.vocab_file='gs://hamishi-dev/easylm/llama/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='hf_prompt' \
    --train_dataset.text_processor.fields='[instruction]' \
    --train_dataset.hf_prompt_dataset.seq_length=64 \
    --train_dataset.hf_prompt_dataset.batch_size=1 \
    --train_dataset.hf_prompt_dataset.num_workers=32 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.init_lr=1e-5 \
    --optimizer.adamw_optimizer.lr=1e-5 \
    --optimizer.adamw_optimizer.end_lr=1e-5 \
    --optimizer.adamw_optimizer.warmup_ratio=0.0 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-PPO-Jax' \
    --logger.prefix='debug' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='wandb' \
    --use_tpu=False \
    --mini_batch_size=1 \
    --max_continuation_len=16 \
    --num_epochs=1 \
    --max_steps_per_epoch=1
