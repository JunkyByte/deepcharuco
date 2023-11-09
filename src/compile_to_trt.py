import torch_tensorrt
import torch
import types
from models.model_utils import speedy_bargmax2d
from inference import load_models

# First load the model and compile it to scripted
dc, ref = load_models('./reference/longrun-epoch=99-step=369700.ckpt',
                      './reference/second-refinenet-epoch-100-step=373k.ckpt', device='cpu')
dc_p = dc.model
ref_p = ref.model

def forward_dc_trt(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = img
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
    cPa = self.relu(self.bnPa(self.convPa(x)))
    loc = self.convPb(cPa)  # NO activ
    cDa = self.relu(self.bnDa(self.convDa(x)))
    ids = self.convDb(cDa)  # NO activ
    return loc, ids

def forward_ref_trt(self, patches: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    x = patches.unsqueeze(1)
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
    cPa = self.relu(self.bnPa(self.convPa(x)))
    loc_hat = self.convPb(cPa).squeeze(1)
    return (speedy_bargmax2d(loc_hat) - 32) / 8 + keypoints

dc_p.forward = types.MethodType(forward_dc_trt, dc_p)
ref_p.forward = types.MethodType(forward_ref_trt, ref_p)

dc_script = torch.jit.script(dc_p.cuda())
ref_script = torch.jit.script(ref_p.cuda())

MAX_BATCH_SIZE = 64

trt_ts_module = torch_tensorrt.compile(dc_script,
    inputs = [
        torch_tensorrt.Input(
            shape=[MAX_BATCH_SIZE, 1, 240, 320], dtype=torch.float32)],
    workspace_size = 1 << 33,
    enabled_precisions = {torch.float32},
)

# Check if it works!
print('Saving!')
torch.jit.save(trt_ts_module, "./reference/deepc_trt.ts") # save the TRT embedded Torchscript
print('Testing')
result = trt_ts_module(torch.zeros([MAX_BATCH_SIZE, 1, 240, 320]).cuda()) # run inference

# Same for refinenet
trt_ts_module = torch_tensorrt.compile(ref_script,
    inputs = [
        torch_tensorrt.Input(
            # min_shape=[1, 1, 24, 24],
            # opt_shape=[8, 1, 24, 24],
            shape=[dc_p.n_ids * MAX_BATCH_SIZE, 24, 24], dtype=torch.float32),
        torch_tensorrt.Input(
            # min_shape=[1, 2],
            # opt_shape=[8, 2],
            shape=[dc_p.n_ids * MAX_BATCH_SIZE, 2], dtype=torch.float32)],
    workspace_size = 1 << 33,
    enabled_precisions = {torch.float32},
)

print('Saving!')
torch.jit.save(trt_ts_module, "./reference/refinenet.ts") # save the TRT embedded Torchscript
print('Testing')
result = trt_ts_module(torch.zeros([dc_p.n_ids * MAX_BATCH_SIZE, 24, 24]).cuda(),
                       torch.zeros((dc_p.n_ids * MAX_BATCH_SIZE, 2)).cuda())  # run inference
