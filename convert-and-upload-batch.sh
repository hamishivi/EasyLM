set -ex

model_list=(
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-safety_20/50bf88ee828942a486cd52a7813e157a/streaming_params_18034,llama_2_7b-tulu_all-safety_20"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-safety_40/b9dbbb7dfeea4c6a9f8db7588c28afda/streaming_params_18860,llama_2_7b-tulu_all-safety_40"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-safety_60/d5497c2f87664295bc57d886fabe2456/streaming_params_19688,llama_2_7b-tulu_all-safety_60"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-safety_80/81f1fd58348f435ebbc201dba91d2128/streaming_params_20514,llama_2_7b-tulu_all-safety_80"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-safety_20/4ba6ea82c1ec46369622eb8e161cfaa4/streaming_params_3304,tulu_2_7b-tulu_match-safety_20"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-safety_40/eb79a418c23c462db30c861479c67a55/streaming_params_3306,tulu_2_7b-tulu_match-safety_40"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-safety_60/b0a056a384b24d098620b250f26319b9/streaming_params_4960,tulu_2_7b-tulu_match-safety_60"
)

for tuple in "${model_list[@]}"
do
IFS=',' read -r MODEL_PATH MODEL_NAME <<< "$tuple"
    mkdir tmp

    # MODEL_PATH=$1
    # MODEL_SIZE=7b
    # MODEL_NAME=$3
    # WORKSPACE=$4

    gsutil cp gs://hamishi-east1/easylm/llama/tokenizer.model tokenizer.model

    python -m EasyLM.models.llama.convert_easylm_to_hf --load_checkpoint=params::${MODEL_PATH} --tokenizer_path='tokenizer.model' --model_size='7b' --output_dir=tmp

    beaker dataset create tmp --name ${MODEL_NAME} --workspace ai2/modular_adaptation &> tmp.log

    # parse beaker id from log. Format: Uploading <name> to <id>
    BEAKER_ID=$(awk '/Uploading/ {print $4}' tmp.log)

    echo  "${MODEL_NAME} uploaded to beaker with id ${BEAKER_ID}"

    # cleanup
    rm -rf tmp
done