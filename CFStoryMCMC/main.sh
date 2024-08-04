
export ROOT_PATH=/home/feiteng/Documents/sources
export DATASET_PATH=${ROOT_PATH}/ctrltextgen/timetravel
export COHERENCE_MODEL_PATH=${DATASET_PATH}/cfstory_nli_metrics/roberta-large

export OUT_DIR=${DATASET_PATH}/test
if [ ! -d "$OUT_DIR" ];then
  mkdir ${OUT_DIR}
  echo "create datadir: "${OUT_DIR}
else
  echo "exists"
fi
python3 counterfactual_rewrite.py \
--data_path ${DATASET_PATH}/test_data_original_end_splitted.json \
--output_dir ${OUT_DIR} \
--output_file ${OUT_DIR}/gpt2medium_robertabase.txt \
--mlm_path roberta-base \
--gpt2_path gpt2-medium \
--causal_token_finding \
--constraint_model_path ${COHERENCE_MODEL_PATH} \
--coherence_type ents \
--gpu_id 0 \
--worker_id 0 \
--num_gpus 4 \
