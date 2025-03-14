from datasets.vindr_mammo import VindrMammoDataset
from omegaconf import OmegaConf
import os
import cv2

DConf = OmegaConf.load('./configs/datasets.yaml')
dataset = VindrMammoDataset(**DConf.Train.VindrMammoDataset)

for i in range(10):
    item = dataset[i]
    os.makedirs('dump', exist_ok=True)
    cv2.imwrite(f'dump/target_{i}.jpg', (item['jpg'] + 1) / 2 * 255)
    cv2.imwrite(f'dump/reference_{i}.jpg', item['ref'] * 255)
    cv2.imwrite(f'dump/hint_{i}.jpg', (item['hint'] + 1) / 2 * 255)
    print(item)
