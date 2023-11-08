import torch_tensorrt
import torch
from inference import load_models

# First load the model and compile it to scripted
dc, ref = load_models('./reference/longrun-epoch=99-step=369700.ckpt',
                      './reference/second-refinenet-epoch-100-step=373k.ckpt', device='cuda')

# dc_script = torch.jit.script(dc.model)
ref_script = torch.jit.script(ref.model)

# trt_ts_module = torch_tensorrt.compile(dc_script,
#     inputs = [
#         torch_tensorrt.Input(
#             min_shape=[1, 1, 144, 192],
#             opt_shape=[1, 1, 240, 320],
#             max_shape=[1, 1, 480, 640], dtype=torch.float)],
#     enabled_precisions = {torch.float},
# )
# 
# # Check if it works!
# print('Testing')
# result = trt_ts_module(torch.zeros([1, 1, 240, 320]).cuda()) # run inference
# print('Saving!')
# torch.jit.save(trt_ts_module, "deepc_trt.ts") # save the TRT embedded Torchscript

# Same for refinenet
# with torch_tensorrt.logging.debug():
trt_ts_module = torch_tensorrt.compile(ref_script,
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1, 24, 24],
            opt_shape=[8, 1, 24, 24],
            max_shape=[16, 1, 24, 24], dtype=torch.float),
        torch_tensorrt.Input(
            min_shape=[1, 2],
            opt_shape=[8, 2],
            max_shape=[16, 2], dtype=torch.float)],
    enabled_precisions = {torch.float},
)

print('Testing')
result = trt_ts_module(torch.zeros([16, 1, 24, 24]).cuda(), torch.zeros((16, 2)).cuda()) # run inference
torch.jit.save(trt_ts_module, "refinenet.ts") # save the TRT embedded Torchscript