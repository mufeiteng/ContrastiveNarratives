import os
import random, torch, numpy
from brownian_bridge_system import BrownianBridgeSystem
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import hydra
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@hydra.main(config_path="config", config_name="brownian_bridge", version_base='1.1')
def run(config):

    print(config)

    seed_everything(
        config.experiment_params.seed,
        use_cuda=config.experiment_params.cuda)
    datatype = config.data_params.name
    name = datatype+'-'+config.data_params.language_encoder+'-dim'+str(config.model_params.latent_dim)
    print(name)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_bridge_loss',
        mode='min',
        filename=name+'-{epoch}-{avg_val_loss:.5f}-{val_bridge_loss:.5f}',
        save_top_k=3,
        dirpath=config.wandb_settings.exp_dir,
        verbose=True,
    )
    wandb_logger = WandbLogger(
        project=config.wandb_settings.project+"-"+name,
        name=config.wandb_settings.exp_name,
        config=config,
    )
    system = BrownianBridgeSystem(config)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0,1,2,3],
        default_root_dir=config.wandb_settings.exp_dir,
        callbacks=[ckpt_callback],
        max_epochs=int(config.experiment_params.num_epochs),
        min_epochs=int(config.experiment_params.num_epochs),
        check_val_every_n_epoch=config.experiment_params.checkpoint_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=wandb_logger
    )

    trainer.fit(system)


if __name__ == "__main__":
    run()



