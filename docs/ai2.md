# Running EasyLM (Hamish Ver.) At AI2

Here's a crash course on running this library. I'll focus on users at AI2, who are currently the primary consumers of this fork. This is a beta of this document, as things may change over time.

## Setup

You'll need a working ai2 account, and be able to login to the [google cloud console](console.cloud.google.com). Under your team project, create a bucket (in cloud storage). Make sure it is in the us region, **MAKE SURE IT IS US-EAST REGION** (to avoid network fees). Then give `972660293965-compute@developer.gserviceaccount.com` read and write permissions to the bucket. This is the TPU account, and will let the TPU read and write from the bucket.

Join the AI2-TPU google group and you should automatically get added to the Ai2 TPU google cloud project (also viewable through the console). This is the project we will make TPUs under, but keep the bucket under your team's project (for accounting purposes).

Download and install the [gcloud CLI](https://cloud.google.com/sdk/gcloud) on your developer device of choice (e.g. your laptop). This will be the primary way we interact with the TPUs. `gcloud auth login` should work for authenticating.

That's it! Now we can setup TPUs.

## Running your first TPU

I try to stick to the command line. Once authenticated, you can make a TPU VM with:
```bash
gcloud alpha compute tpus tpu-vm create <name> --accelerator-type=v3-8 --zone=us-central1-a --project=ai2-tpu --version=v2-alpha
```
Or, if you want a bigger TPU:
```bash
gcloud alpha compute tpus tpu-vm create <name> --accelerator-type=v3-256 --zone=us-east1-d --project=ai2-tpu --version=v2-alpha
```

Note the different `zone`s, this is because of how our quota works. I usually name my tpus something like `hamishi-v3-8` (i.e., username-tpu-size). We only have access to v3s and v2s right now, and note that the larger the TPU you want, the higher the likelihood is you can't get it immediately and might have a try a few times (ask me about this). Sometimes there are simply no TPUs available in the research cloud.

Now, the way you interact with the TPUs is going to be different depending on the size:

For a v3-8, you can just ssh directly into the TPU with `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-central1-a --project=ai2-tpu`. Then just treat it like your development machine: clone repositories, run code, whatever! This is useful for debugging on a TPU.

For anything larger, the TPU is actually a collection of machines (1 per 8 cores). Jax can handle the coordination between the machines... so long as the environment between them is identical. So in this case, we can send single ssh commands to all machines like so: `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="<command>"`
For this reason, it's useful to have scripts that do what you want to avoid having to type lots of commands.

You can quickly and easily test everything is working by running a little Jax program:
```bash
gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="python3 -c \"import jax; jax.devices()\""
```

Anyway, now we are set up! If you need to delete a TPU, you can use the `gcloud alpha compute tpus tpu-vm delete` command.

**NOTE: It is important to note that for all the commands below, you should be running them on all workers with `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command=` if you are using a TPU pod (anything over 8 cores). Yes, this means running the git clone command, setup script, and train command on multiple machines.**

### Using Someone Else's TPU

Sometimes it's useful to jump on someone else's TPU instead of yours (for example, if they are not using it). Only one person can use a TPU at a time, so please coordinate with the person as to who is using it. Sometimes the previous user may have left lockfiles that cause permission errors. You can fix these by removing the lockfiles:
```bash
sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs
```

And again, you should be running this via `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command=` if you are using a TPU pod.

## Running EasyLM

I'll omit the TPU ssh command bits here, but remember to use them for a TPU pod!

```bash
git clone https://github.com/hamishivi/easylm.git  # and swap to whatever branch you want
cd easylm; ./scripts/tpu_vm_setup.sh  # setup environment
wandb login <key>  # for logging into wandb
gsutil -m cp gs://hamishi-east1/easylm/data/tulu-v2-sft-mixture.jsonl .  # download whatever data you want. TODO: I need to allow data to be streamed...
```

And that's it! Then you can go nuts and train. Here's an example *finetuning* command:
```bash
cd easylm; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,8' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/7b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='tulu_json_torch' \
    --train_dataset.json_torch_dataset.hf_name='allenai/tulu-v2-sft-mixture' \
    --train_dataset.json_torch_dataset.hf_split='train' \
    --train_dataset.json_torch_dataset.seq_length=8192 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.entity='rlhf-llm-dev' --logger.project='jax_sft' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &
```

Note a few things:
- most things load directly from the google bucket! And you can load datasets from huggingface, so long as they follow the same format as `allenai/tulu-v2-sft-mixture` (or `allenai/ultrafeedback_binarized_cleaned` for preference data). Alternatively, you can instead specify `train_dataset.json_torch_dataset.path` to point to a file either on the TPU or in a bucket (e.g. `train_dataset.json_torch_dataset.path='gs://hamishi-east1/data/...`).
- there's a bunch of scary random TPU args, these are just args I found that people recommended. I haven't properly tested them...
- the `mesh_dim` defines the parallelism strategy. Check out the EasyLM parallelism doc for more information. Generally, you want the biggest FSDP parallelism (middle number), and smallest model parallelism possible (last number). The numbers must multiply to the TPU size (e.g. 256 for v3-256).
- Currently I am lazy and just download the datafile to the TPU
- Edit the logger to point to whatever wandb project you want. Entity is the group I've been putting this stuff in.
- `output_dir` is where the model gets saved
- `&> all.log &` is there so the output gets streamed to `all.log`, and then the process runs in the background, to avoid needing constant ssh connection. There may be a less hacky way to do this. You can inspect the output with `tail -f all.log`.

Generally, I recommend waiting and checking for the model to start logging steps, and then you are probably fine.

Here's an example dpo train script, and note that it is basically the same:
```bash
cd easylm; git pull; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_dpo \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=3 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/tulu2_7b_fixed/263f4f758b194729b206d5adad2b50d7/streaming_params_20384' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-7 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.1 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='preference_json_torch' \
    --train_dataset.json_torch_dataset.hf_name='allenai/ultrafeedback_binarized_cleaned' \
    --train_dataset.json_torch_dataset.hf_split='train_prefs' \
    --train_dataset.json_torch_dataset.seq_length=6144 \
    --train_dataset.json_torch_dataset.batch_size=8 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='easylm_dpo' --logger.entity='rlhf-llm-dev' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &"
```

I have arguments for the beta, etc, but these are not used here. Other example scripts live in the `examples/` directory.

### PPO training

Here's an example PPO training script:
```bash
cd easylm; git pull; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_ppo \
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
    --logger.entity='liujch1998' \ # Remember to change this to your WANDB entity
    --logger.project='n-Tulu-PPO-Jax' \ # Remember to change this to your WANDB project
    --logger.prefix='train_v3.2_v3_interleave-fwd-bwd_nofreeze' \ # Bump this version number for each run. Format: train_{new_version}_{old_version}_{describe_the_diff}
    --logger.prefix_to_id=True \
    --logger.wandb_dir='/home/jiachengl/wandb' \ # Remember to change this to your TPU's local directory
    --logger.output_dir='gs://jiachengl-east1/n-tulu-ppo-jax/' \ # You may keep using this, or change to your GCS bucket
    --use_tpu=True \
    --ppo_epochs=1 \
    --lr=1e-6 \
    --kl_coef=0.05 \
    --reward_gain=1.0 --reward_bias=0.0 \
    --save_milestone_freq=116 \
    --num_epochs=1 \
    &> /home/jiachengl/all.log &" # Remember to change this to your TPU's local directory
```

You may also refer to `examples/ppo_13b_tpu.sh`, this is the script I use to start jobs.
I set an environment variable `WANDB_API_KEY` with the value on my AI2 server, if you went through the standard `wandb login` process, you may remove that command.

**How to set hyperparameters:**
There are a few hyperparameters that are somewhat related and you need to watch out for constaints.
If you are not sure what values to use, you can use the default values below.
* **mesh_dim**: This is a tuple of three integers, corresponding to the (DP, FSDP, MP) dimensions. The product of these three numbers should be equal to the number of TPUs. Please always set the DP dimension to 1; I noticed that otherwise it will mess up with the rollouts, and I couldn't figure out a solution. Default: (1, 64, 4).
* **prompt_batch_size** ($B_p$), **rollouts_per_prompt** ($r$): $B_p$ is the batch size with which the prompt dataset is chunked into, and each batch of prompts correspond to one "step" in the WANDB log. Each prompt is rolled out $r$ times. $B_c$ (**completion_batch_size**) is the effective batch size after the rollout, and is automatically set to $B_c = B_p \times r$. Default: $B_p = 512, r = 1$.
* **forward_mini_batch_size** ($B_f$), **backward_mini_batch_size** ($B_b$): These are the batch sizes used for rollouts, forward passes, and backward passes. $B_f$ and $B_b$ must each be both a divisor of $B_c$ and a multiple of (DP x FSDP); I usually set them to be equal to (DP x FSDP) and haven't tested otherwise. Default: $B_f = B_b = 64$.
* Note: as of v3.2, forward_mini_batch_size and backward_mini_batch_size are merged into a single hparam, **mini_batch_size**.
* **gradient_accumulation_steps** ($g$): The number of backward passes before a gradient update is done. Typically I'd say make $B_b \times g \le B_c$. Default: $g = 1$.
* **ppo_epochs** ($e$): The number of inner epochs performed on each batch of completions. Default: $e = 1$.
* **How many batches of prompts are there in an epoch?** Say the length of prompt dataset is $D$. Then the answer is $D / B_p$.
* **How many gradient updates are done?** For each batch of prompts, the number of gradient updates is $(B_c \times e) / (B_b \times g)$. For one full epoch, the number of gradient updates is $(D \times r \times e) / (B_b \times g)$.
* **warmup_epochs**: The number of epochs during which LR is being linearly warmed up. Default: 0.1.
<!-- * **policy_freeze_epochs**: The number of epochs to wait before policy LR starts warming up. The purpose is to let the value model train for a period and converge, so that the policy doesn't train on garbage value estimates. Default: 0.5. -->

A few notes:
* **RM prompt format:** By default we assume that the RM is trained to take the same format as in Tulu. If you use the original `UltraRM-13b`, you need to customize the RM prompt format by adding these options:
```
    --train_dataset.hf_prompt_dataset.reward_prefix_tokens='Human: ' \
    --train_dataset.hf_prompt_dataset.reward_suffix_tokens='\nAssistant: ' \
```
* **Time:** On a v3-256 TPU, training a 13b-13b model for 1 epoch (with the default hparams above) takes about 7 minutes per iteration, and a full epoch (~120 batches) takes about 16 hours.

## Killing a job

I usually use `pgrep llama | xargs kill -9` to ensure everything is dead.

## Preference data max sequence length for running data abltaion
```
orca_dpo_pairs: 2368
helpsteer: 2730
ultrafeedback_mean_aspects_cleaned: 6100
ultrafeedback_overall_cleaned: 6100
anthropic_hh.jsonl: 4787 (>4096: 59)
nectar.jsonl: 10732 (>4096: 34)
shp_2.jsonl: 52667 (>4096: 8402)
stack_exchange_paired.jsonl: 89209 (>4096: 51317)
```

# Exporting and evaluting your models

Now you've trained a model, what to do? I'll walk you through the steps now, but also note I have a script that runs these steps automatically (although knowing the steps will help with debugging)!

## 0.1 Setup beaker

This is a one-time thing - make sure you have a beaker workspace setup, and then make sure there is a secret set in the beaker workspace called `OPENAI_API_KEY` with your OpenAI API key set (this will be used for running evaluation) - see [the beaker docs for more](https://beaker-docs.apps.allenai.org/concept/secrets.html).

## 1. Install EasyLM on a server

First, clone and install this repo (i.e., run `conda env create -f scripts/gpu_environment.yml`). You won't need the GPUs for exporting, so you could just modify the install for CPU-only too. I recommend doing this on a cirrascale machine so you have a fast internet connection, lots of storage space, and lots of RAM for downloading and converting the models, but I guess you could run this locally.

## 2. Download tokenizer

You'll need the `tokenizer.model` file for the model you are converting. For most Llama models (1/2), you can download mine by running `gsutil cp gs://hamishi-dev/easylm/llama/tokenizer.model tokenizer.model` (you might have to authenticate with `gcloud auth application-default login` if you haven't used gsutil on this machine before).

For other models (e.g. Codellama), you can find the tokenizer usually in the huggingface page. Specifically, you'll want the file called `tokenizer.model`.

## 3. Exporting to HF format

Identify the final checkpoint in your google bucket - look in the folder you saved to with `logger.output_dir`, then look for `streaming_param_<some number>`, it'll be the largest number - it saves every epoch. Then convert with:
```bash
python -m EasyLM.models.llama.convert_easylm_to_hf --load_checkpoint=params::<path> --tokenizer_path=<where you downloaded the tokenizer to> --model_size=<model_size> --output_dir=<output_dir>
```
`model_size` is just the size of the model - look at `LLAMA_STANDARD_CONFIGS` in `EasyLM.models.llama.convert_easylm_to_hf` if you want a list. For your output dir, put it somewhere local.

## 4. (optional) Upload to beaker

Optionally, you can upload the model to beaker datasets, to avoid storing all your checkpoints locally. You can do this with:
```bash
beaker dataset create <model_save_dir> --name <model_name> --workspace ai2/<your_workspace>
```

I recommend giving the model a name that you can easily remember and distinguish from others :)
Note the beaker id that is output (under `saving to...`) - you'll need this for the next step.

## 5. Run eval scripts

Finally, you can run the `open-instruct` eval suite by running:
```bash
python scripts/submit_open_instruct_eval.py --workspace <your_workspace> --model_name <model_name> --location <beaker_id_or_nfs_path> --num_gpus 1 --is_tuned
```

If you didn't upload your model to beaker, alternatively you can give a path to a directory on cirrascale NFS. The model name is what the beaker jobs will be called, so pick something easy to distinguish! Additionally, for models larger than 13B, you might want to set more GPUs per job.

If you need the evaluation jobs to finish quickly, try specifying a cluster (e.g. `--cluster ai2/allennlp-cirrascale`) and using a non-preemptible priority (e.g. `--priority high`).

If you want to edit how the beaker jobs are run in some way, look at `beaker_configs/default_eval.yaml` - this is what the jobs are patterned off. For example, you might want to change the image the jobs run on.

## 6. Sit back and relax

Thats it! Once the jobs are running, it should take ~1 hour for the evaluations to run (for a 7B model). Once done, record them down wherever you are recording them down :)

## ALT: Run the script

As opposed to the above, you can also just run the following script, which just runs through the above steps automatically. You will still need a beaker workspace setup with `OPENAI_API_KEY` set/
Then:
```bash
./scripts/convert_and_submit_tuned MODEL_PATH MODEL_SIZE MODEL_NAME WORKSPACE
```

You'll need to fill in the four variables above with the requisite bits - see the previous steps for what these should be.

*Note: this script assumes a llama tokenizer, which isn't true for anything that isnt llama 1/2. For those, you'll have to do the conversion + evaluation steps more manually (but trust me, it's easy work!)*.

# Random other tips

## Debugging

Sometimes you can get inscrutable errors on TPUs. Here is a rough guide to fixing them:
1. Inspect the logs *on all TPU workers*. Sometimes an error on one worker means other TPUs 'randomly' crash. For example, one TPU runs out of space, or the master TPU can't login to wandb, or even just a random error.
2. Ensure all processes are dead with `pgrep llama | xargs kill -9` and `ps -aux`. If there are other accelerator-using processes, starting new ones will crash.
3. Run `gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="python3 -c \"import jax; print(jax.devices())\""`. If this exits and prints all the devices, then your TPU is probably fine and I think you should look carefully at your logs and code for issues. If not (and there are no other TPU processes running in the background), then your TPU is probably in a bad state and you need to delete and recreate it.
