from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
import configs
from data_refinenet import RefineDataset
from models.refinenet import RefineNet, lRefineNet
import pytorch_lightning as pl


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)
    total = 14

    dataset = RefineDataset(config, config.train_labels, config.train_images,
                            visualize=False, validation=False, total=total)

    dataset_val = RefineDataset(config, config.val_labels, config.val_images,
                                visualize=False, validation=True, total=total)

    bs_train = max(1, config.bs_train_rn // total)
    bs_val = max(1, config.bs_val_rn // total)
    print(f'Using virtual train batch size of {bs_train} images')
    print(f'Using virtual val batch size of {bs_val} images')
    train_loader = DataLoader(dataset, batch_size=bs_train,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True, prefetch_factor=10)
    val_loader = DataLoader(dataset_val, batch_size=bs_val,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=True, prefetch_factor=10)

    model = RefineNet()
    train_model = lRefineNet(model)

    logger = TensorBoardLogger("tb_logs", name="refinenet")
    checkpoint_callback = ModelCheckpoint(dirpath="tb_logs/ckpts_refinenet/", save_top_k=10,
                                          monitor="val_refinenet_loss")
    trainer = pl.Trainer(max_epochs=60, logger=logger, accelerator="auto",
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint='./reference/epoch=27-step=122668.ckpt')
    trainer.fit(train_model, train_loader, val_loader)
