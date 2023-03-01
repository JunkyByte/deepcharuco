from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
import configs
from data import CharucoDataset
from models.net import lModel, dcModel
import pytorch_lightning as pl


if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)

    # TODO setup datasets correctly using CONFIG for paths
    dataset = CharucoDataset(config,
                             config.train_labels,
                             config.train_images,
                             visualize=False,
                             validation=False)

    # TODO: Might need to take a subset of this one
    dataset_val = CharucoDataset(config,
                                 config.val_labels,
                                 config.val_images,
                                 visualize=False,
                                 validation=True)

    train_loader = DataLoader(dataset, batch_size=config.batch_size_train,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True, prefetch_factor=5)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size_val,
                            shuffle=False, num_workers=config.num_workers,
                            pin_memory=True, prefetch_factor=5)

    model = dcModel(n_ids=config.n_ids)
    train_model = lModel(model)

    logger = TensorBoardLogger("tb_logs", name="deepcharuco")
    trainer = pl.Trainer(max_epochs=100, logger=logger, accelerator="auto",
                         callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
                         resume_from_checkpoint='./tb_logs/deepcharuco/version_2/checkpoints/epoch=22-step=85031.ckpt')
    trainer.fit(train_model, train_loader, val_loader)
