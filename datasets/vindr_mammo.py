import cv2
import numpy as np
import os
import pandas as pd

from .data_utils import *
from .base import BaseDataset

class VindrMammoDataset(BaseDataset):
    def __init__(
        self,
        image_dir: str,
        anno_path: str,
        split: str,
        image_size: int = 512,
        aug_ref: bool = True,
        random_erase_prob: float = 0.0,
        anomaly_type: str = "reinsert",  # "reinsert", "replace", "insert"
        channels: int = 1,
    ):
        """
        Args:
            image_dir (str): The directory of the png images.
            anno_path (str): The path to the annotation file.
            split (str): The split of the dataset.
            image_size (int): The size of the image context and target.
            aug_ref (bool): Whether to augment the reference image.
            random_erase_prob (float): The probability of random erase mask.
            anomaly_type (str): Type of setting to use - "reinsert", "replace", or "insert"
        """
        self.image_dir = image_dir
        self.split = split
        self.image_size = image_size
        self.aug_ref = aug_ref
        self.random_erase_prob = random_erase_prob
        self.anomaly_type = anomaly_type
        self.channels = channels

        self.data = pd.read_csv(anno_path)
        self.data[['xmin', 'xmax', 'ymin', 'ymax']] = self.data[['xmin', 'xmax', 'ymin', 'ymax']].clip(lower=0)
        
        self.healthy_data = self.data[
            (self.data["split"] == self.split) &
            (self.data["finding_categories"] == "[\'No Finding\']")
        ].reset_index(drop=True)
        self.data = self.data[
            (self.data["split"] == self.split) &
            (self.data["finding_categories"] != "[\'No Finding\']")
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def _rate_insertion_bbox(self, bbox, binary_mask, width, height):
        """Check if at least 90% of the bbox area is within the foreground and dimensions are within 15% of target."""
        xmin, ymin, xmax, ymax = bbox

        if xmax <= xmin or ymax <= ymin:
            return 0

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        if not (width * 0.8 <= bbox_width <= width * 1.2 and 
                height * 0.8 <= bbox_height <= height * 1.2):
            return 0.5

        bbox_area = bbox_width * bbox_height
        foreground_pixels = np.sum(binary_mask[ymin:ymax, xmin:xmax])
        return foreground_pixels / bbox_area

    def _get_valid_insertion_bbox(self, target_image, binary_mask, width, height, max_attempts=20):
        """Find a valid bbox for insertion that meets the foreground coverage requirement."""
        best_bbox = None
        best_rating = 0
        
        for _ in range(max_attempts):
            non_background = np.where(target_image != np.argmax(np.bincount(target_image.flatten())))
            rand_idx = np.random.randint(0, len(non_background[0]))
            xmin = non_background[1][rand_idx]
            ymin = non_background[0][rand_idx]
            
            xmax = min(xmin + width, target_image.shape[1])
            ymax = min(ymin + height, target_image.shape[0])
            
            bbox = np.array([xmin, ymin, xmax, ymax]).astype(np.int32)
            rating = self._rate_insertion_bbox(bbox, binary_mask, width, height)
            if rating > 0.9:
                return bbox
            
            if rating >= best_rating:
                best_rating = rating
                best_bbox = bbox

        if best_rating == 0:
            print(f"No valid bbox found for {width} x {height}")
            print(num_attempts)
            return bbox
        
        # Return the best bbox found, or the last attempted one if no valid bbox was found
        return best_bbox

    def get_sample(self, idx):
        sample_id = self.data.iloc[idx]['study_id'] + '_' + self.data.iloc[idx]['image_id']

        # Target image
        is_erase_mask = np.random.rand() < self.random_erase_prob
        if is_erase_mask or self.anomaly_type == "insert":
            target_annomaly_data = self.healthy_data.iloc[np.random.randint(0, len(self.healthy_data))]
        else:
            target_annomaly_data = self.data.iloc[idx]
        target_image_path = os.path.join(self.image_dir, target_annomaly_data['study_id'], target_annomaly_data['image_id'] + '.png')
        target_image = cv2.imread(target_image_path) # H x W x 3

        # Target bbox
        if is_erase_mask or self.anomaly_type == "insert":
            # Sample a bbox within the target image on foreground
            background_value = np.argmax(np.bincount(target_image.flatten()))
            binary_mask = (target_image != background_value).astype(np.uint8)
            if self.anomaly_type == "insert":
                binary_mask = cv2.erode(binary_mask, np.ones((20, 20), np.uint8), iterations=1) # removes text
                width =  self.data.iloc[idx]['xmax'] - self.data.iloc[idx]['xmin']
                height = self.data.iloc[idx]['ymax'] - self.data.iloc[idx]['ymin']
            else:
                widths = self.data['xmax'] - self.data['xmin']
                heights = self.data['ymax'] - self.data['ymin']
                width = int(np.random.choice(widths))
                height = int(np.random.choice(heights))
            
            max_attempts = 20 if self.anomaly_type == "insert" else 1
            target_bbox = self._get_valid_insertion_bbox(target_image, binary_mask, width, height, max_attempts)
        else:
            target_bbox = np.array([target_annomaly_data['xmin'], target_annomaly_data['ymin'], target_annomaly_data['xmax'], target_annomaly_data['ymax']]).astype(np.int32)
        
        target_bbox_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
        target_bbox_mask[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]] = 1

        # Source bbox and reference image
        if self.anomaly_type == "insert" or self.anomaly_type == "replace":
            if self.anomaly_type == "insert":
                source_anomaly_data = self.data.iloc[idx]
            else:
                # Filter anomalies with similar width and height (±15%) to the target bbox
                target_width = target_bbox[2] - target_bbox[0]
                target_height = target_bbox[3] - target_bbox[1]
                similar_width_mask = (self.data['xmax'] - self.data['xmin']).between(
                    target_width * 0.85, target_width * 1.15
                )
                similar_height_mask = (self.data['ymax'] - self.data['ymin']).between(
                    target_height * 0.85, target_height * 1.15
                )
                
                # Randomly select one anomaly from similar sized anomalies, excluding current index
                similar_size_indices = self.data[similar_width_mask & similar_height_mask].index.tolist()
                similar_size_indices = [i for i in similar_size_indices if i != idx]
                rng = np.random.RandomState(idx)  # Create a local random number generator for reproducibility
                if similar_size_indices:
                    selected_idx = rng.choice(similar_size_indices)
                    source_anomaly_data = self.data.iloc[selected_idx]
                else:
                    print(f"No similar sized anomalies found for {sample_id}")
                    print(f"Size of target anomaly: {target_bbox[2] - target_bbox[0]} x {target_bbox[3] - target_bbox[1]}")
                    fallback_indices = [i for i in range(len(self.data)) if i != idx]
                    source_anomaly_data = self.data.iloc[rng.choice(fallback_indices)]

            source_image_path = os.path.join(self.image_dir, source_anomaly_data['study_id'], source_anomaly_data['image_id'] + '.png')
            source_image = cv2.imread(source_image_path)
            source_bbox = np.array([source_anomaly_data['xmin'], source_anomaly_data['ymin'], source_anomaly_data['xmax'], source_anomaly_data['ymax']]).astype(np.int32)
            source_bbox_mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
            source_bbox_mask[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]] = 1
        else:
            source_bbox = target_bbox
            source_bbox_mask = target_bbox_mask
            source_image = target_image

        reference_image = source_image[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]].copy()
        reference_mask = source_bbox_mask[source_bbox[1]:source_bbox[3], source_bbox[0]:source_bbox[2]].copy()

        # Process source and target data
        item_with_collage = self.process_pairs(
            reference_image,
            reference_mask,
            target_image,
            target_bbox_mask,
            crop_ratio=[3.0, 4.0],
            image_size=self.image_size,
            channels=self.channels,
            aug_ref=self.aug_ref,
            is_erase_mask=is_erase_mask,
        )
        timestep = np.random.randint(0, 1000)

        item_with_collage['time_steps'] = np.array([timestep])
        item_with_collage['sample_id'] = sample_id
        item_with_collage['original_image_path'] = target_image_path
        item_with_collage['resized_reference_image'] = cv2.resize(reference_image, (224, 224))

        return item_with_collage
