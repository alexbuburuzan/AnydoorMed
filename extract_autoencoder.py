import os
import torch
from omegaconf import OmegaConf
from cldm.model import create_model, load_state_dict

config = OmegaConf.load('./configs/inference_orig.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cpu'), strict=False)

state_dict = model.first_stage_model.state_dict()
os.makedirs("checkpoints/autoencoder/", exist_ok=True)
torch.save({"state_dict": state_dict}, "checkpoints/autoencoder/anydoor_image_vae.ckpt")