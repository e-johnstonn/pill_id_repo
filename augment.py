import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2




image_paths = os.listdir('')


transform_rotate = A.Compose([
    A.Rotate(limit=[90, -90], p=1, border_mode=cv2.BORDER_CONSTANT),
    ToTensorV2()
])

transform_distort = A.Compose([
    A.Perspective(scale=(0.05, 0.1)),
    ToTensorV2()
])

transform_brightness_contrast = A.Compose([
    A.RandomBrightnessContrast(p=1),
    ToTensorV2()
])

transforms = {'rotate': transform_rotate, 'distort': transform_distort,
              'brightness_contrast': transform_brightness_contrast}

counter = 1

for image_path in image_paths:
    image_path = os.path.join('pill_outputs', image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        continue
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    for transform_name, transform in transforms.items():
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = transformed_image.permute(1, 2, 0).numpy().astype('uint8')

        base_filename = os.path.basename(image_path)  # get just the filename
        base, ext = os.path.splitext(base_filename)
        new_filename = f"aug/{base}_{transform_name}_{counter}{ext}"

        cv2.imwrite(new_filename, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

    counter += 1



