from torch import optim, nn
from models.model_utils import pre_bgr_image, pred_argmax
import torch
import numpy as np
import pytorch_lightning as pl


class RefineNet(torch.nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        c1, c2, c3, c4 = 64, 64, 128, 128
        self.last_c = 8
        det_h = 4096
        self.reBn = True

        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(c4)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, self.last_c, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(self.last_c)

        # self.convPb = torch.nn.Conv2d(self.last_c, det_h, kernel_size=1, stride=1, padding=0)
        self.out = torch.nn.Linear(self.last_c * 3 * 3, det_h)

    def forward(self, x):
        """
        Input
          x: Image pytorch tensor shaped N x 1 x 24 x 24.
        """

        # Let's stick to this version: first BN, then relu
        x = self.relu(self.bn1a(self.conv1a(x)))
        conv1 = self.relu(self.bn1b(self.conv1b(x)))
        x, ind1 = self.pool(conv1)
        x = self.relu(self.bn2a(self.conv2a(x)))
        conv2 = self.relu(self.bn2b(self.conv2b(x)))
        x, ind2 = self.pool(conv2)
        x = self.relu(self.bn3a(self.conv3a(x)))
        conv3 = self.relu(self.bn3b(self.conv3b(x)))
        x, ind3 = self.pool(conv3)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))

        # Head
        cPa = self.relu(self.bnPa(self.convPa(x)))
        loc = self.out(cPa.view((-1, self.last_c * 3 * 3)))  # NO activ
        return loc

    def infer_image(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Inference on BGR image, assuming no pre processing

        Parameters
        ----------
        img : np.ndarray
            The bgr image
        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            loc, ids output
        """
        raise NotImplementedError
        with torch.no_grad():
            img = pre_bgr_image(img)
            img = torch.tensor(np.expand_dims(img, axis=0))
            loc_hat, ids_hat = self(img).values()
            loc_hat = loc_hat.cpu().numpy()
            ids_hat = ids_hat.cpu().numpy()
        return pred_argmax(loc_hat, ids_hat, dust_bin_ids=self.n_ids)


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  padding=(kernel_size - 1) // 2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


# define the LightningModule
class lRefineNet(pl.LightningModule):
    def __init__(self, dcModel):
        super().__init__()
        self.model = dcModel

    def forward(self, x):
        return self.model(x)

    def infer_image(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.model.infer_image(img)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, loc = batch.values()
        loc_hat = self.model(x).values()

        loss_loc = nn.functional.cross_entropy(loc_hat, loc)

        self.log("val_refinenet_loss", loss_loc)
        return loss_loc, loss_ids

    def training_step(self, batch, batch_idx):
        x, loc = batch.values()
        loc_hat = self.model(x).values()

        loss_loc = nn.functional.cross_entropy(loc_hat, loc)

        self.log("train_refinenet_loss", loss_loc)
        return loss_loc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=4e-3)
        return optimizer


if __name__ == '__main__':
    model = RefineNet()

    # from torchinfo import summary
    from torchinfo import summary
    summary(model, input_size=(1, 1, 24, 24))
