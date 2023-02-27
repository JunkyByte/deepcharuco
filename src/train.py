from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
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
                              shuffle=True, num_workers=configs.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=configs.batch_size_val,
                            shuffle=False, num_workers=configs.num_workers, pin_memory=True)

    model = dcModel(n_ids=configs.n_ids)
    train_model = lModel(model)

    logger = TensorBoardLogger("tb_logs", name="deepcharuco")
    trainer = pl.Trainer(limit_train_batches=25, limit_val_batches=25,
                         max_epochs=1, logger=logger)
    trainer.fit(train_model, train_loader, val_loader)
