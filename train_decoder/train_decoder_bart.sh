export ROOT_DIR=_datapath
export encoder_dir=${ROOT_DIR}/brownian_bridge/encoder_training
export decoder_dir=${ROOT_DIR}/brownian_bridge/decoder_training
export encoder_checkpoint_path=bbencoder.ckpt


export dim=64
export lr=5e-5
export ratio=0.15
export expri_name=dim${dim}_lr${lr}_ratio${ratio}
echo ${expri_name}

python load_cl_feats.py \
--train_path ${ROOT_DIR}/rocstories/rocstories_train_split.json \
--val_path ${ROOT_DIR}/rocstories/rocstories_dev_split.json \
--test_path ${ROOT_DIR}/rocstories/rocstories_test_split.json \
--model_type 'bart' \
--output_path 'rocstories_event2feats.pickle' \
--latent_dim 64 \
--model_name_or_path "facebook/bart-base" \
--datatype "rocstories" \
--encoder_filepath ${encoder_checkpoint_path} \
--per_gpu_eval_batch_size 32 \
--num_workers 8 \
--gpu_id 0 \


accelerate launch train_decoder_bart.py \
--train_path ${ROOT_DIR}/rocstories/rocstories_train_split_contrastive_excepted.json \
--val_path ${ROOT_DIR}/rocstories/rocstories_dev_split.json \
--test_path ${ROOT_DIR}/rocstories/rocstories_test_split.json \
--output_dir ${decoder_dir}/${expri_name} \
--model_type 'bart' \
--feat_path 'rocstories_event2feats.pickle' \
--model_name_or_path "facebook/bart-base" \
--datatype "rocstories" \
--latent_dim ${dim} \
--perturb_ratio ${ratio} \
--encoder_filepath ${encoder_checkpoint_path} \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--source_len 128 \
--target_len 96 \
--num_workers 8 \
--learning_rate ${lr} \
--weight_decay 1e-5 \
--warmup_ratio 0.1 \
--logging_steps 100 \
--validate_steps 500 \
--num_train_epochs 5 \
--use_contrastive_embeddings \
--do_train \
