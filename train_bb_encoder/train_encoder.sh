
export hidden_dims=(32)
export model_types=("bart")

export experi_name=bart-ft-bbencoder
echo ${experi_name}
#nohup
python train_encoder.py \
--config-name=brownian_bridge \
data_params.name=rocstories \
data_params.language_encoder="bart" \
model_params.latent_dim=64 \
experiment_params.num_epochs=20 \
optim_params.batch_size=128 \
