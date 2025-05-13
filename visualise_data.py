import argparse, os
import cv2
import einops
import numpy as np
import torch
import random
from ldm.util import instantiate_from_config
from pytorch_lightning import seed_everything
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
from datasets.base import BaseDataset
from datasets.vindr_mammo import VindrMammoDataset

conf = OmegaConf.load("configs/anydoor.yaml")
data_config = conf["data"]["params"]["test"]

dataset = VindrMammoDataset(**data_config["params"])

save_dir = "dump"
os.makedirs(save_dir, exist_ok=True)

for i in range(10):
    sample = dataset[i]

    ref_image = sample["ref"]
    target_image = sample["jpg"]
    hint_image = sample["hint"]

    print(i)
    # min max normalize
    ref_image = (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min()) if ref_image.max() != ref_image.min() else ref_image
    target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min()) if target_image.max() != target_image.min() else target_image
    hint_image = (hint_image - hint_image.min()) / (hint_image.max() - hint_image.min()) if hint_image.max() != hint_image.min() else hint_image

    # convert to uint8
    ref_image = (ref_image * 255).astype(np.uint8)
    target_image = (target_image * 255).astype(np.uint8)
    hint_image = (hint_image * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_dir, f"{i}_ref.png"), ref_image)
    cv2.imwrite(os.path.join(save_dir, f"{i}_target.png"), target_image)
    cv2.imwrite(os.path.join(save_dir, f"{i}_hint.png"), hint_image[..., [0]])
    # cv2.imwrite(os.path.join(save_dir, f"{i}_original.png"), original_image)