import os
import time
import torch

from typing import Dict, Union, Optional
from torch.nn import Module
from transformers import AutoModel

last_gc = 0


def torch_gc():
    global last_gc
    if time.time() - last_gc > 60:
        last_gc = time.time()
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"Emptying gpu cache {device}...")
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        if num_gpus > torch.cuda.device_count():
            raise Exception(
                f"need {num_gpus} GPU, but only has {torch.cuda.device_count()}")

        from accelerate import dispatch_model
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model
