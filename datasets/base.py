import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A


class BaseDataset(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    @staticmethod
    def aug_data_back(image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    @staticmethod
    def aug_data_mask(image, mask, aug_ref = False):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        if aug_ref:
            transformed = transform(image=image.astype(np.uint8), mask = mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
        else:
            transformed_image = image.copy()
            transformed_mask = mask.copy()
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.data)-1)
        item = self.get_sample(idx)
        return item
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        return ratio > 0.8 * 0.8  and ratio < 0.1 * 0.1

    @staticmethod
    def process_pairs(
        ref_image: np.ndarray,
        ref_mask: np.ndarray,
        tar_image: np.ndarray,
        tar_mask: np.ndarray,
        max_ratio: float = 0.8,
        crop_ratio: float = [1.3, 3.0],
        mask_ratio: float = [1.1, 1.5],
        image_size: int = 512,
        channels: int = 3,
        aug_ref: bool = False,
        is_erase_mask: bool = False,
    ) -> dict:
        """
        Process the reference and target images and masks to prepare for training.

        Args:
            ref_image (np.ndarray): The reference image, size H x W x 3, uint8, [0, 255]
            ref_mask (np.ndarray): The reference mask, size H x W, uint8, 0 BG, 1 FG 
            tar_image (np.ndarray): The target image, size H x W x 3, uint8, [0, 255]
            tar_mask (np.ndarray): The target mask, size H x W, uint8, 0 BG, 1 FG
            max_ratio (float): The maximum ratio of the target mask area to the reference mask area.
            crop_ratio (float): Controls the size of the target image crop.
            image_size (int): The size of the image context and target.
            channels (int): The number of channels of the image.
            aug_ref (bool): Whether to augment the reference image.
            is_erase_mask (bool): Whether to erase the mask.

        Returns:
            dict: A dictionary containing the processed reference and target images and masks.
        """
        # assert mask_score(ref_mask) > 0.90
        # assert self.check_mask_area(ref_mask) == True
        # assert self.check_mask_area(tar_mask)  == True

        # ========= Reference ===========
        '''
        # similate the case that the mask for reference object is coarse. Seems useless :(

        if np.random.uniform(0, 1) < 0.7: 
            ref_mask_clean = ref_mask.copy()
            ref_mask_clean = np.stack([ref_mask_clean,ref_mask_clean,ref_mask_clean],-1)
            ref_mask = perturb_mask(ref_mask, 0.6, 0.9)
            
            # select a fake bg to avoid the background leakage
            fake_target = tar_image.copy()
            h,w = ref_image.shape[0], ref_image.shape[1]
            fake_targe = cv2.resize(fake_target, (w,h))
            fake_back = np.fliplr(np.flipud(fake_target))
            fake_back = BaseDataset.aug_data_back(fake_back)
            ref_image = ref_mask_clean * ref_image + (1-ref_mask_clean) * fake_back
        '''

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask) # (h_min, h_max, w_min, w_max), tuple
        # assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min') == True
        
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1) # H_ref x W_ref x 3
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) # white BG

        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        ref_mask = ref_mask[y1:y2,x1:x2]

        ratio = np.random.randint(mask_ratio[0] * 10, mask_ratio[1] * 10) / 10 # equivalent to original implementation
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) # max_expanded_size x max_expanded_size x 3
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8) # 224 x 224 x 3

        ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False) # max_expanded_size x max_expanded_size x 3 
        ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8) # 224 x 224 x 3
        ref_mask = ref_mask_3[:,:,0] # 224 x 224

        # Augmenting reference image
        # masked_ref_image_aug = self.aug_data(masked_ref_image) 
        
        # Getting for high-freqency map
        masked_ref_image_compose, ref_mask_compose = BaseDataset.aug_data_mask(masked_ref_image, ref_mask, aug_ref = aug_ref) 
        masked_ref_image_aug = masked_ref_image_compose.copy()
        ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1) # 224 x 224 x 3, {0, 255}
        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255) # 224 x 224 x 3, uint8, [0, 255]
        

        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        # assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        
        # Cropping around the target object 
        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=crop_ratio)   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy() 
        if is_erase_mask:
            ref_image_collage *= 0
        try:
            collage[y1:y2,x1:x2,:] = ref_image_collage
        except:
            pass

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2,x1:x2,:] = 1.0
        collage_bbox_mask = collage_mask.copy()

        if np.random.uniform(0, 1) < 0.7: 
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        collage_bbox_mask = pad_to_square(collage_bbox_mask, pad_value = 2, random = False).astype(np.uint8)

        H2, W2 = collage.shape[0], collage.shape[1]
        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (image_size,image_size)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (image_size,image_size)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (image_size,image_size),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_bbox_mask = cv2.resize(collage_bbox_mask.astype(np.uint8), (image_size,image_size),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1
        collage_bbox_mask[collage_bbox_mask != 1] = 0
        
        # Prepairing dataloader items
        masked_ref_image_aug = masked_ref_image_aug  / 255 
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0 

        if is_erase_mask:
            masked_ref_image_aug *= 0

        if channels == 1:
            cropped_target_image = cropped_target_image[..., [0]]
            collage = collage[..., [0]]

        collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

        # extract box from collage_bbox_mask
        reference_box_yyxx = get_bbox_from_mask(collage_bbox_mask)
        
        item = dict(
                ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                reference_box_yyxx=np.array(reference_box_yyxx)
                ) 
        return item





