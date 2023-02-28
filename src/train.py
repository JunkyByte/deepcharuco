from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
import configs
from data import CharucoDataset
from models.net import lModel, dcModel
import pytorch_lightning as pl


if __name__ == '__main__':
    configs = load_configuration(configs.CONFIG_PATH)

    # TODO setup datasets correctly using CONFIG for paths
    dataset = CharucoDataset(configs,
                             configs.train_labels,
                             configs.train_images,
                             visualize=False,
                             validation=False)

    # TODO: Might need to take a subset of this one
    dataset_val = CharucoDataset(configs,
                                 configs.val_labels,
                                 configs.val_images,
                                 visualize=False,
                                 validation=True)

    train_loader = DataLoader(dataset, batch_size=configs.batch_size_train,
                              shuffle=True, num_workers=configs.num_workers,
                              pin_memory=True, prefetch_factor=5)
    val_loader = DataLoader(dataset_val, batch_size=configs.batch_size_val,
                            shuffle=False, num_workers=configs.num_workers,
                            pin_memory=True, prefetch_factor=5)

    model = dcModel(n_ids=configs.n_ids)
    train_model = lModel(model)

    logger = TensorBoardLogger("tb_logs", name="deepcharuco")
    trainer = pl.Trainer(max_epochs=100, logger=logger, accelerator="auto",
                         callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])#,
                         #ckpt_path='./tb_logs/deepcharuco/version_0/checkpoints/epoch=0-step=3697.ckpt')
    trainer.fit(train_model, train_loader, val_loader)
