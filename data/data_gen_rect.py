import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pydicom


def check_side(image):
    slice_l = image[:, :100]
    slice_r = image[:, -100:]
    mean_l = slice_l.mean()
    mean_r = slice_r.mean()
    if mean_l > mean_r:
        return 'L'
    elif mean_r > mean_l:
        return 'R'
    else:
        print('Error: Laterality could not be determined!')
        return 'E'


def extract_roi(roi: str):  # convert ROI string to list
    roi = roi.translate({ord(c): None for c in "][)(,"})
    roi = list(map(int, roi.split()))
    roi_list = []
    for i in range(len(roi) // 4):
        roi_list.append(roi[4*i:4*i+4])
    return roi_list


def save_scan(npy, img_dir, filename):
    # Save array
    np.save(img_dir+filename[:-4]+'.npy', npy)


def get_mask(img, roi_list: list):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for roi in roi_list:
        mask[roi[0]:roi[2], roi[1]:roi[3]] = 1.0
    return mask


def save_dataset(mode: str, image_rows: int = 1024, image_cols: int = 832, base_path: str = ''):
    if mode == 'train':
        df_path = '/home/careinfolab/unet_mammo/images/pos_train.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm/train/'
    elif mode == 'test':
        df_path = '/home/careinfolab/unet_mammo/images/pos_test.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm/test/'
    else:
        df_path = '/home/careinfolab/unet_mammo/images/pos_val.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm/val/'

    df = pd.read_csv(df_path)
    for i, data in tqdm(df.reset_index().iterrows()):
        img_npy = np.ndarray((image_rows, image_cols, 2), dtype=np.float32)
        dcm = pydicom.dcmread(base_path + data.anon_dicom_path[26:])
        img = dcm.pixel_array

        roi_list = extract_roi(data.ROI_coords)
        filename = data.png_filename

        # correct image laterality
        img_side = check_side(img)
        if img_side == 'R':
            img = np.fliplr(img)
        elif img_side == 'E':
            continue

        mask = get_mask(img, roi_list)

        # resize image and mask
        img = cv.resize(img, dsize=(image_cols, image_rows), interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, dsize=(image_cols, image_rows), interpolation=cv.INTER_AREA)

        # normalize image between 0 and 1
        img = img.astype(np.float32)
        img_norm = (img-np.min(img))/(np.max(img)-np.min(img))

        # Save the image to channel 0 and the mask to channel 1
        img_npy[:, :, 0] = img_norm
        img_npy[:, :, 1] = mask

        save_scan(img_npy, img_dir, filename)
    print(f"'{mode}' set saved...\n")


save_dataset('train', base_path='/media/careinfolab/CI_Lab')
save_dataset('val', base_path='/media/careinfolab/CI_Lab')
save_dataset('test', base_path='/media/careinfolab/CI_Lab')