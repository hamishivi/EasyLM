set -ex

model_list=(
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_with_coding-coding_20/98a737fa602d497e856e093867ef6c50/streaming_params_40828,llama_2_7b-tulu_all_with_coding-coding_20"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_with_coding-coding_40/5235e100f0764902831ab35875c37181/streaming_params_22370,llama_2_7b-tulu_all_with_coding-coding_40"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_with_coding-coding_60/35a84ced077044048a4442a8ffa06082/streaming_params_24326,llama_2_7b-tulu_all_with_coding-coding_60"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_with_coding-coding_80/f1c63875b5ec475ebaa5ee54e6ecacec/streaming_params_26284,llama_2_7b-tulu_all_with_coding-coding_80"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_match-coding_20/71b53b25a48c48f1957da4d6d014bb05/streaming_params_3912,tulu_2_7b_with_coding-tulu_match-coding_20"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_match-coding_40/3a3b886aa57a4fe2bcc38b680b93bdca/streaming_params_7824,tulu_2_7b_with_coding-tulu_match-coding_40"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_match-coding_60/8f6ab65d127047fb9793ab7984909717/streaming_params_11736,tulu_2_7b_with_coding-tulu_match-coding_60"
    # "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b_with_coding-tulu_match-coding_80/f56c99abf12d4d2e84b922ac1a4efca2/streaming_params_15648,tulu_2_7b_with_coding-tulu_match-coding_80"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_no_science_no_safety_no_coding-science_2500-safety_100-coding_100/e5001a8f46fa442883e98ffdb7641564/streaming_params_34958,llama_2_7b-tulu_all-science_2500-safety_100-coding_100"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/llama_2_7b-tulu_all_no_science_no_safety_no_coding-science_2500-safety_100/9848cfef147c4074b57dea534025aaac/streaming_params_12588,llama_2_7b-tulu_all-science_2500-safety_100"
    "gs://jacobm-bucket/modular_adaptation/checkpoints/consistent_mix/tulu_2_7b-tulu_none-science_2500-safety_100-coding_100/eacdac9faa40440e885da5cc57223bfc/streaming_params_8874,tulu_2_7b-tulu_none-science_2500-safety_100-coding_100"
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