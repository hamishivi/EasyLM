gcloud alpha compute tpus tpu-vm ssh jiachengl-v3-512 --zone=us-east1-d --project=ai2-tpu --worker=all --command="git clone https://github.com/liujch1998/EasyLM.git n-tulu-ppo-jax; cd n-tulu-ppo-jax; git checkout ppo; ./scripts/tpu_vm_setup.sh"
