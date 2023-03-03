from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
import configs
from data import CharucoDataset
from models.net import lModel, dcModel
import pytorch_lightning as pl


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)

    dataset = CharucoDataset(config,
                             config.train_labels,
                             config.train_images,
                             visualize=False,
                             validation=False)

    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    train_loader = DataLoader(dataset, batch_size=config.bs_train,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True, prefetch_factor=10)
    val_loader = DataLoader(dataset_val, batch_size=config.bs_val,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=True, prefetch_factor=10)

    model = dcModel(n_ids=config.n_ids)
    train_model = lModel(model)

    logger = TensorBoardLogger("tb_logs", name="deepcharuco")
    checkpoint_callback = ModelCheckpoint(dirpath="tb_logs/ckpts_deepcharuco/", save_top_k=10,
                                          monitor="val_loss")
    trainer = pl.Trainer(max_epochs=60, logger=logger, accelerator="auto",
                         callbacks=[checkpoint_callback]) #,
                         # resume_from_checkpoint='./reference/epoch=44-step=83205.ckpt')
    trainer.fit(train_model, train_loader, val_loader)
