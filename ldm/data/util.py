import torch
import cv2
import numpy as np

from ldm.modules.midas.api import load_midas_transform


class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample


def draw_contour(
    images: np.ndarray,
    masks: np.ndarray,
    color: tuple = (1.0, 0.5, 0.0)
) -> np.ndarray:
    """
    Draw contours around `masks` on `images`

    Parameters:
    images (numpy.ndarray): Shape (N, 3, 512, 512), values in range [-1, 1]
    masks (numpy.ndarray): Shape (N, 1, 512, 512), binary masks where 1 is foreground
    color (tuple): RGB color for the contours, default is Matplotlib orange (1.0, 0.5, 0.0)

    Returns:
    numpy.ndarray: Images with contours, shape (N, 3, 512, 512), values in range [-1, 1]
    """
    batch_size = masks.shape[0]
    result = images.copy()
   
    for i in range(batch_size):
        img_hwc = np.transpose(result[i], (1, 2, 0)).astype(np.float32).copy()
        img_normalized = (img_hwc + 1) / 2.0

        mask = (masks[i][0] == 1).astype(np.uint8)

        # Find and draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_normalized, contours, -1, color, 2)

        img_result = img_normalized * 2.0 - 1.0
        result[i] = np.transpose(img_result, (2, 0, 1))

    return result