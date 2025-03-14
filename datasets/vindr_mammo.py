import cv2
import numpy as np
import os
import pandas as pd

from .data_utils import *
from .base import BaseDataset

class VindrMammoDataset(BaseDataset):
    def __init__(self, image_dir, anno_path, split):
        self.image_dir = image_dir
        self.split = split

        self.data = pd.read_csv(anno_path)
        self.data = self.data[
            (self.data["split"] == self.split) &
            (self.data["finding_categories"] != "[\'No Finding\']")
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        annomaly_data = self.data.iloc[idx]
        target_image_path = os.path.join(self.image_dir, annomaly_data['study_id'], annomaly_data['image_id'] + '.png')
        target_image = cv2.imread(target_image_path) # H x W x 3

        # get bbox from xmin, ymin, xmax, ymax
        target_bbox = np.array([annomaly_data['xmin'], annomaly_data['ymin'], annomaly_data['xmax'], annomaly_data['ymax']]).astype(np.int32)
        target_bbox_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
        target_bbox_mask[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = 1

        # get reference image and mask
        source_bbox = target_bbox
        reference_image = target_image[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]].copy()
        reference_mask = target_bbox_mask[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]].copy()  # Create copy to avoid view

        item_with_collage = self.process_pairs(reference_image, reference_mask, target_image, target_bbox_mask, crop_ratio=[3.0, 4.0])
        timestep = np.random.randint(0, 1000) # avoid dynamic sampling for image and video
        item_with_collage['time_steps'] = np.array([timestep])

        return item_with_collage
