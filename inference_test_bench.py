import argparse, os
import cv2
import einops
import numpy as np
import torch
import random
from ldm.util import instantiate_from_config, move_to_device
from pytorch_lightning import seed_everything
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from cldm.model import load_state_dict

from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
from datasets.base import BaseDataset
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--copy_paste",
        action="store_true",
    )
    parser.add_argument(
        'overrides',
        nargs=argparse.REMAINDER,
        help='Configuration overrides',
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(opt.config)
    cli_conf = OmegaConf.from_dotlist(opt.overrides)
    config = OmegaConf.merge(config, cli_conf)

    if not opt.copy_paste:
        model = instantiate_from_config(config.model)
        model.load_state_dict(load_state_dict(opt.ckpt, location=device), strict=False)
        model = model.to(device).eval()
        ddim_sampler = DDIMSampler(model)
    
    batch_size = opt.batch_size
    num_workers = opt.n_workers

    test_data_config = config.data.params.test
    test_dataset = instantiate_from_config(test_data_config)

    test_dataloader= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=opt.n_workers, 
        pin_memory=True, 
        shuffle=False,
        drop_last=False
    )

    pred_path = os.path.join(opt.outdir, 'pred')
    os.makedirs(pred_path, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            sample_id = batch['sample_id']
            batch = move_to_device(batch, device)
            
            if save_memory and not opt.copy_paste:
                model.low_vram_shift(is_diffusing=False)

            ref = batch['ref']
            hint = batch['hint']
            num_samples = ref.shape[0]

            if save_memory and not opt.copy_paste:
                model.low_vram_shift(is_diffusing=True)

            if not opt.copy_paste:
                control = einops.rearrange(hint, 'b h w c -> b c h w').float()
                ref_control = einops.rearrange(ref, 'b h w c -> b c h w').float()

                guess_mode = False
                H,W = 512,512
                strength = 1
                eta = 0.0

                cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( ref_control )]}
                un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
                shape = (4, H // 8, W // 8)

                model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                samples, intermediates = ddim_sampler.sample(
                    opt.ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=eta,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=un_cond)

                if save_memory:
                    model.low_vram_shift(is_diffusing=False)

                x_samples = model.decode_first_stage(samples)
                x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            target_all = (batch['jpg'] * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            context_all = (batch["hint"] * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            ref_all = (batch['ref'] * 255).cpu().numpy().astype(np.uint8)
            ref_all_resized = batch["resized_reference_image"].cpu().numpy().astype(np.uint8)

            if opt.copy_paste:
                x_samples = context_all[..., [0]]

            for i in range(num_samples):
                patch_pred = x_samples[i].mean(-1)[..., None]
                patch_target = target_all[i][..., [0]]
                anomaly_ref = ref_all[i][..., [0]]
                anomaly_ref_resized = ref_all_resized[i][..., [0]]
                sizes = batch['extra_sizes'][i]
                context_collage = context_all[i][..., [0]]
                context_mask = context_all[i][..., [-1]]
                tar_box_yyxx_crop = batch['tar_box_yyxx_crop'][i]
                reference_box_yyxx = batch['reference_box_yyxx'][i]
                y1, y2, x1, x2 = reference_box_yyxx

                if opt.copy_paste:
                    pad = 2
                    xmin, xmax = max(0, x1-pad), min(512, x2+pad)
                    ymin, ymax = max(0, y1-pad), min(512, y2+pad)
                    patch_pred[ymin:ymax, xmin:xmax] = cv2.resize(anomaly_ref_resized, (xmax - xmin, ymax - ymin))[..., None]

                grid = np.concatenate([
                    context_collage,
                    cv2.resize(anomaly_ref, context_collage.shape[:2])[..., None],
                    patch_pred,
                    patch_target], axis=1)

                pred_reference = patch_pred[y1:y2, x1:x2]
                pred_reference = cv2.resize(pred_reference, (224, 224))

                # Define output directories and their corresponding data
                output_dirs = {
                    "patch_pred": patch_pred,
                    "patch_target": patch_target,
                    "anomaly_ref_padded": anomaly_ref,
                    "anomaly_ref_resized": anomaly_ref_resized,
                    "context_collage": context_collage,
                    "context_mask": context_mask,
                    "pred_reference": pred_reference,
                    "grid": grid
                }

                # Create directories and save images
                for dir_name, image_data in output_dirs.items():
                    dir_path = os.path.join(pred_path, dir_name)
                    os.makedirs(dir_path, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dir_path, f'{sample_id[i]}_{dir_name}.jpg'),
                        image_data
                    )