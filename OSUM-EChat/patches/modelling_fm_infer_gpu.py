import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import f5_tts
from f5_tts.model.backbones.dit_mask import DiT as DiT_

_GPU_FM_TORCH_COMPILE = True

class GPUDiT(DiT_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_forward = torch.compile(self.fast_forward, dynamic=False, fullgraph=True) \
            if _GPU_FM_TORCH_COMPILE else self.fast_forward

# ===================================================================
print("========================= DO FM PATCH ============================")
# ===================================================================
f5_tts.model.backbones.dit_mask.DiT = GPUDiT