import os


import torch
import numpy as np
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
from scipy.ndimage import center_of_mass




model = FastSAM('weights/FastSAM-x.pt')

image_dir = ''

images_list = os.listdir(image_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def slice_image(image_path, masks, output_size=512, fill_value=0):
    image = np.array(Image.open(image_path))
    sliced_images = []
    half_size = output_size // 2
    for mask in masks:
        mask = mask.detach().cpu().numpy().astype(bool)

        centroid_y, centroid_x = center_of_mass(mask)
        centroid_y, centroid_x = int(round(centroid_y)), int(round(centroid_x))
        padding_y = (max(half_size - centroid_y, 0), max(half_size - (image.shape[0] - centroid_y), 0))
        padding_x = (max(half_size - centroid_x, 0), max(half_size - (image.shape[1] - centroid_x), 0))

        if image.ndim == 3:
            image_padded = np.pad(image, (padding_y, padding_x, (0, 0)), mode='constant', constant_values=fill_value)
        else:
            image_padded = np.pad(image, (padding_y, padding_x), mode='constant', constant_values=fill_value)


        mask_padded = np.pad(mask, (padding_y, padding_x), mode='constant')

        start_y, start_x = centroid_y + padding_y[0] - half_size, centroid_x + padding_x[0] - half_size
        end_y, end_x = centroid_y + padding_y[0] + half_size, centroid_x + padding_x[0] + half_size

        if image_padded.ndim == 3:
            window = image_padded[start_y:end_y, start_x:end_x, :]
        else:
            window = image_padded[start_y:end_y, start_x:end_x]

        window[~mask_padded[start_y:end_y, start_x:end_x]] = fill_value

        sliced_images.append(window)

    return sliced_images


def save_image(tensor, file_path):
    image = Image.fromarray(tensor)
    image.save(file_path)



for image in images_list:
    try:
        everything = model(os.path.join(image_dir, image), device=device, retina_masks=True, imgsz=1024, conf=.93, iou=.9)
    except Exception as e:
        print(e)
        print('No objects detected.')
        continue
    prompt_process = FastSAMPrompt(os.path.join(image_dir, image), everything, device=device)
    ann = prompt_process.everything_prompt()

    if len(ann) > 0:
        sliced_images = slice_image(os.path.join(image_dir, image), ann)

        for i, sliced_image in enumerate(sliced_images):
            save_image(sliced_image, f'./pill_outputs/{image}_{i}.jpg')
