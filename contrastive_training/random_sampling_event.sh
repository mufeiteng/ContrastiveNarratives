export PJ_PATH=/home/feiteng/Documents/sources/ctrltextgen
export CLOZE_DIR=${PJ_PATH}/cloze_datasets
export COPA_DIR=${PJ_PATH}/copa_datasets
export ANLI_DIR=${PJ_PATH}/anli_datasets
export ECARE_DIR=${PJ_PATH}/e-CARE/dataset/Causal_Reasoning
export Cfstory_DIR=${PJ_PATH}/timetravel
export HellaSwag_DIR=${PJ_PATH}/hellaswag_datasets
export Swag_DIR=${PJ_PATH}/swag_datasets

export model_size=large

export num_con_samples=(15 7)
#export num_epochs=(5)
#export trade_off_weights=(0.5)
export tradeoff=0.5
export num_epoch=5
export negsample_end_choices=(20 40 60)
export perturb_ratio_choices=(0.1 0.15)
#timetravel20k_constexample_dim64_ratio0.2_nobridge_nomask_numgenerated_merged.jsonl
for negsample_end in "${negsample_end_choices[@]}" ;
do
  for perturb_ratio in "${perturb_ratio_choices[@]}" ;
  do
    for num_sample in "${num_con_samples[@]}" ;
    do

      export EXPER_TYPE=tt_20k_bb_nucles_ratio${perturb_ratio}_roberta${model_size}_numsample${num_sample}_negsample_end${negsample_end}_tradeoff${tradeoff}_coheval
      export OUTPUT_DIR=${PJ_PATH}/MCMC_discriminator/${EXPER_TYPE}
  #--train_data_file ${PJ_PATH}/timetravel/train_supervised_large_original_end_split.json \
  #    --train_data_file ${PJ_PATH}/timetravel/train_supervised_large_sampled_20k.jsonl \

      #CUDA_VISIBLE_DEVICES=4,5,6,7
      #nohup accelerate launch --main_process_port 20688
      CUDA_VISIBLE_DEVICES=0,5,6,7 nohup accelerate launch --main_process_port 20688 random_sampling_event.py \
      --train_data_file ${PJ_PATH}/timetravel/timetravel20k_constexample_dim64_ratio${perturb_ratio}_nuclessampling_merged_kept70.jsonl \
      --eventsim_dict ${PJ_PATH}/timetravel/timetravel_event_cossimilarity.json \
      --negsample_start 0 \
      --negsample_end ${negsample_end} \
      --trade_off ${tradeoff} \
      --num_sample ${num_sample} \
      --model_type roberta \
      --model_size ${model_size} \
      --output_dir ${OUTPUT_DIR} \
      --copa_data_dir ${COPA_DIR} \
      --ecare_data_dir ${ECARE_DIR} \
      --anli_data_dir ${ANLI_DIR} \
      --cloze_data_dir ${CLOZE_DIR} \
      --cfstory_data_dir ${Cfstory_DIR} \
      --hellaswag_data_dir ${HellaSwag_DIR} \
      --swag_data_dir ${Swag_DIR} \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 1 \
      --per_gpu_eval_batch_size 16 \
      --num_workers 2 \
      --learning_rate 5e-5 \
      --weight_decay 1e-4 \
      --warmup_ratio 0.1 \
      --max_grad_norm 1.0 \
      --logging_steps 1000 \
      --validate_steps 4000 \
      --num_train_epochs ${num_epoch} \
      --do_train \
      --gpu_id 0 \
      > run_${EXPER_TYPE}.log 2>&1 &


      wait
    done

  done
done