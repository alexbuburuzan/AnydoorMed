import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
from datasets.base import BaseDataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


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


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = BaseDataset.process_pairs(
        ref_image,
        ref_mask,
        tar_image,
        tar_mask,
        mask_ratio=[1.2, 1.3],
        channels=1,

    )
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']

    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image


if __name__ == '__main__': 
    # ==== Example for inferring a single image ===
    reference_image_path = './examples/TestDreamBooth/FG/01.png'
    bg_image_path = './examples/TestDreamBooth/BG/000000047948_GT.png'
    bg_mask_path = './examples/TestDreamBooth/BG/000000047948_mask.png'
    save_path = './examples/TestDreamBooth/GEN/gen_res.png'

    # reference image + reference mask
    # You could use the demo of SAM to extract RGB-A image with masks
    # https://segment-anything.com/demo
    image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
    mask = (image[:,:,-1] > 128).astype(np.uint8)
    image = image[:,:,:-1]
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    ref_image = image 
    ref_mask = mask

    # background image
    back_image = cv2.imread(bg_image_path).astype(np.uint8)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # background mask 
    tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
    tar_mask = tar_mask.astype(np.uint8)
    
    gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    h,w = back_image.shape[0], back_image.shape[0]
    ref_image = cv2.resize(ref_image, (w,h))
    vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    cv2.imwrite(save_path, vis_image [:,:,::-1])
    '''
    # ==== Example for inferring VITON-HD Test dataset ===

    from omegaconf import OmegaConf
    import os 
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './VITONGEN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    image_names = os.listdir(test_dir)
    
    for image_name in image_names:
        ref_image_path = os.path.join(test_dir, image_name)
        tar_image_path = ref_image_path.replace('/cloth/', '/image/')
        ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
        tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        gt_image = cv2.imread(tar_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 5

        gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
        gen_path = os.path.join(save_dir, image_name)

        vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
        cv2.imwrite(gen_path, vis_image[:,:,::-1])
    '''

    

