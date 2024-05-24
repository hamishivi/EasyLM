set -ex

model_list=(
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_with_coding-coding_100/5f465534920b4adeb4b1f1fa1ed42ae9/streaming_params_28240","llama_2_7b-tulu_all_with_coding-coding_100"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_none-coding_20/79b17b9a825c437296b565bb20af6569/streaming_params_1956","tulu_2_7b_with_coding-tulu_none-coding_20"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_none-coding_40/2793006888f345e5af376be1993730da/streaming_params_3912","tulu_2_7b_with_coding-tulu_none-coding_40"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_none-coding_60/c3ac556241e74817bde519c413152c0b/streaming_params_5868","tulu_2_7b_with_coding-tulu_none-coding_60"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_none-coding_80/9d5dc7c0d59c408fa7307bd1a4dbf513/streaming_params_7824","tulu_2_7b_with_coding-tulu_none-coding_80"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_none-coding_100/73a29bd3573a4b529c1577ae4e497544/streaming_params_9782,tulu_2_7b_with_coding-tulu_none-coding_100"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_match-coding_100/103f170e99b3417ebb470f9f0274032e/streaming_params_19560,tulu_2_7b_with_coding-tulu_match-coding_100"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-science_100/060ee7c60a97401ab0eb37c332b01998/streaming_params_17466,llama_2_7b-tulu_all-science_100"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-science_200/be7d91b23b844f1aa03c122614e8e8f8/streaming_params_17720,llama_2_7b-tulu_all-science_200"
    # ",llama_2_7b-tulu_all-science_500"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all-coding_1000/736414108d884b2c9e3cfaf1a8c3d387/streaming_params_19416,llama_2_7b-tulu_all-science_1000"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-science_100/b9f62d9f859241b98dd87d54c9382e61/streaming_params_516,tulu_2_7b-tulu_match-science_100"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-coding_200/6783b23abf4b4e31a712c53e7f1e554a/streaming_params_1022,tulu_2_7b-tulu_match-science_200"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-science_500/98d9de082f464c7f8fa527ef4b14107b/streaming_params_4922,tulu_2_7b-tulu_match-science_500"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_match-science_1000/daa2fd7e0d7043f0a8347ac225bb9501/streaming_params_4418,tulu_2_7b-tulu_match-science_1000"
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