set -ex

mkdir tmp

MODEL_PATH=$1
MODEL_SIZE=$2
MODEL_NAME=$3
WORKSPACE=$4

gsutil cp gs://hamishi-dev/easylm/llama/tokenizer.model tokenizer.model

python -m EasyLM.models.llama.convert_easylm_to_hf --load_checkpoint=params::${MODEL_PATH} --tokenizer_path='tokenizer.model' --model_size=${MODEL_SIZE} --output_dir=tmp

beaker dataset create tmp --name ${MODEL_NAME} --workspace ai2/${WORKSPACE} &> tmp.log

# parse beaker id from log. Format: Uploading <name> to <id>
BEAKER_ID=$(awk '/Uploading/ {print $4}' tmp.log)

# python scripts/submit_open_instruct_eval.py --workspace ${WORKSPACE} --model_name ${MODEL_NAME} --location ${BEAKER_ID} --num_gpus 1 --is_tuned

# echo  "${MODEL_NAME} uploaded to beaker with id ${BEAKER_ID} and submitted to open instruct eval. Check your beaker experiments for the results!"

# cleanup
rm -rf tmp
