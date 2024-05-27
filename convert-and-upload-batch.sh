set -ex

model_list=(
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-coding_20/dcf31467524d4d92805185985fb86054/streaming_params_19164,llama_2_7b-tulu_all-coding_20"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-coding_40/65c07cec8f67468f9c000512ffafcd2c/streaming_params_21120,llama_2_7b-tulu_all-coding_40"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-coding_60/f08ddd32d84a4e1a993b953a468347c5/streaming_params_23076,llama_2_7b-tulu_all-coding_60"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-coding_80/e29c30ebdd9a4306b4f874fee788e85a/streaming_params_25034,llama_2_7b-tulu_all-coding_80"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-coding_20/8bf9279c80e54e5482d08cc9d956588e/streaming_params_7822,tulu_2_7b-tulu_match-coding_20"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-coding_40/21529fc8082c4707a56c2a20fafc7e63/streaming_params_15648,tulu_2_7b-tulu_match-coding_40"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-coding_60/14c4aeb80eba4212a20c46c30a029a03/streaming_params_11736,tulu_2_7b-tulu_match-coding_60"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-coding_80/a11285e5fae14f838acc37c0e0153f15/streaming_params_15648,tulu_2_7b-tulu_match-coding_80"
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