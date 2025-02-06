import os
import pydicom
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import argparse

from typing import Tuple
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def convert_dicom_to_png(study_id: str, image_id: str, data_path: str) -> Tuple[bool, str]:
    """
    Convert a DICOM file to a PNG image

    Args:
        study_id: The study ID
        image_id: The image ID
        data_path: The path to the data directory

    Returns:
        - True if the conversion was successful
        - False if the conversion failed
    - The path to the Dicom file

    """
    dicom_file = f'{data_path}/images/{study_id}/{image_id}.dicom'
    png_file = f'{data_path}/png_images/{study_id}/{image_id}.png'

    data = pydicom.read_file(dicom_file)
    if ('WindowCenter' not in data) or\
       ('WindowWidth' not in data) or\
       ('PhotometricInterpretation' not in data) or\
       ('RescaleSlope' not in data) or\
       ('PresentationIntentType' not in data) or\
       ('RescaleIntercept' not in data):

        logging.warning(f"{dicom_file} DICOM file does not have required fields")
        return (False, dicom_file)

    intentType = data.data_element('PresentationIntentType').value
    if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
        logging.warning(f"{dicom_file} got processing file")
        return (False, dicom_file)

    c = data.data_element('WindowCenter').value # data[0x0028, 0x1050].value
    w = data.data_element('WindowWidth').value  # data[0x0028, 0x1051].value
    if type(c)==pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    try:
        a = data.pixel_array
    except:
        logging.warning(f'{dicom_file} Cannot get get pixel_array!')
        return (False, dicom_file)

    slope = data.data_element('RescaleSlope').value
    intercept = data.data_element('RescaleIntercept').value
    a = a * slope + intercept

    try:
        pad_val = data.get('PixelPaddingValue')
        pad_limit = data.get('PixelPaddingRangeLimit', -99999)
        if pad_limit == -99999:
            mask_pad = (a==pad_val)
        else:
            if str(photometricInterpretation) == 'MONOCHROME2':
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        logging.warning(f'{dicom_file} has no PixelPaddingValue')
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                logging.warning(f'{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}')
        except:
            logging.warning(f'{dicom_file} most frequent pixel value {sorted_pixels[0]}')

    # apply window
    mm = c - 0.5 - (w-1)/2
    MM = c - 0.5 + (w-1)/2
    a[a<mm] = 0
    a[a>MM] = 255
    mask = (a>=mm) & (a<=MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w-1) + 0.5) * 255

    if str( photometricInterpretation ) == 'MONOCHROME1':
        a = 255 - a

    os.makedirs(os.path.dirname(png_file), exist_ok=True)
    saved = cv2.imwrite(png_file, a.astype(np.uint8))
    return (saved, dicom_file)


def convert_dicom_to_png_wrapper(args):
    """ Wrapper to unpack arguments for imap """
    return convert_dicom_to_png(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="data/vindr-mammo",
                      help='Path to data directory containing breast-level_annotations.csv')
    parser.add_argument('--num-cores', type=int, default=1,
                      help='Number of CPU cores to use for processing')
    args = parser.parse_args()

    # Use specified number of CPU cores
    num_processes = args.num_cores
    logging.info(f"Using {num_processes} processes")

    # Read CSV file and create list of parameters for each mammography scan
    df = pd.read_csv(f"{args.data_path}/breast-level_annotations.csv")
    params = zip(df['study_id'], df['image_id'], [args.data_path] * len(df))
    
    # Create pool and process files
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(convert_dicom_to_png_wrapper, params),
            total=len(df),
            desc="Converting DICOM to PNG"
        ))

    # check for failed conversions
    failed = [r for r in results if not r[0]]
    logging.info(f"Failed to convert {len(failed)} out of {len(results)} files")
    for f in failed:
        logging.warning(f"Failed to convert {f[1]}")
    logging.info("Done!")
    