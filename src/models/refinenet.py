from torch import optim, nn
from model_utils import pre_bgr_image, speedy_bargmax2d
import torch
import numpy as np
import pytorch_lightning as pl


class RefineNet(torch.nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        c1, c2, c3, c4, c5 = 64, 128, 128, 128, 64
        self.last_c = 64
        self.reBn = True

        self.up_sample = torch.nn.UpsamplingNearest2d(scale_factor=2)

        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=0)
        self.bn1a = nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=0)
        self.bn1b = nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
        self.bn2a = nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=0)
        self.bn2b = nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(c4)

        self.conv5a = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bn5a = nn.BatchNorm2d(c5)
        self.conv5b = torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        self.bn5b = nn.BatchNorm2d(c5)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c5, self.last_c, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(self.last_c)

        self.convPb = torch.nn.Conv2d(self.last_c, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Input
          x: Image pytorch tensor shaped N x 1 x 24 x 24.
        """

        # Let's stick to this version: first BN, then relu
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))

        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))

        x = self.pool(x)

        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))

        x = self.up_sample(x)

        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))

        x = self.up_sample(x)

        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))

        x = self.up_sample(x)

        # Head
        cPa = self.relu(self.bnPa(self.convPa(x)))
        loc = self.convPb(cPa)


        return loc

    def infer_patches(self, patches: np.ndarray,
                      keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Inference on 24x24 patches, assuming no pre processing

        Parameters
        ----------
        patches : np.ndarray
            The patches (N, 24, 24) or (N, 1, 24, 24)
        keypoints : np.ndarray
            The keypoints corresponding to the patches in (x, y) format
        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            corners in 8x resolution (in original image)
        """
        with torch.no_grad():
            if patches.ndim == 3:
                patches = np.expand_dims(patches, axis=1)
            patches = torch.tensor(patches)
            loc_hat = self(patches)
            loc_hat = loc_hat[:, 0, ...].cpu().numpy()

        # loc_hat: (N, H/8, W/8)
        # TODO: Better to use an actual heatmap peaks extractor?
        corners = speedy_bargmax2d(loc_hat)

        # Add keypoints to center on keypoints, divide by 8 to account for 8x resolution
        # Remove half left window to correct position TODO: check me (consider argmax output)
        corners = (corners - 32) / 8 + keypoints  # Is 32 right? or 31 / 33 :)
        return corners


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
    def __init__(self, refinenet):
        super().__init__()
        self.model = refinenet

    def forward(self, x):
        return self.model(x)

    def infer_patches(self, patches: np.ndarray,
                      keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.model.infer_patches(patches, keypoints)

    def validation_step(self, batch, batch_idx):
        x, loc = batch
        x = torch.stack(x, dim=0)
        loc = torch.stack(loc, dim=0)

        x = x.view(-1, *x.size()[-3:])
        loc = loc.view(-1, *loc.size()[-3:])
        loc_hat = self.model(x)

        loss_loc = nn.functional.mse_loss(loc_hat, loc)

        self.log("val_refinenet_loss", loss_loc)
        return loss_loc

    def training_step(self, batch, batch_idx):
        x, loc = batch
        x = torch.stack(x, dim=0)
        loc = torch.stack(loc, dim=0)

        x = x.view(-1, *x.size()[-3:])
        loc = loc.view(-1, *loc.size()[-3:])
        loc_hat = self.model(x)

        loss_loc = nn.functional.mse_loss(loc_hat, loc)

        self.log("train_refinenet_loss", loss_loc)
        return loss_loc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


if __name__ == '__main__':
    model = RefineNet()

    # from torchinfo import summary
    from torchinfo import summary
    summary(model, input_size=(1, 1, 24, 24))

    # print(model.infer_patches(np.random.randn(16, 24, 24).astype(np.float32)))
