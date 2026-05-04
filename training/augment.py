import albumentations as A
import cv2
import os
from pathlib import Path

transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.ColorJitter(p=0.3),
])

input_dir = Path("data/raw")          # source folder with raw images                               
output_dir = Path("data/processed")   # destination for augmented images                            
num_augmented = 2                    # how many augmented copies per image 

for class_dir in input_dir.iterdir(): # loop: data/raw/pan/, data/raw/cheque/                       
    if not class_dir.is_dir():        # skip if it's a file, not a folder                           
        continue                                                                                 

    save_dir = output_dir / class_dir.name  # e.g., data/processed/pan/
    save_dir.mkdir(parents=True, exist_ok=True)  # create folder if doesn't exist

    for img_path in class_dir.iterdir():    # loop through each image in pan/
        image = cv2.imread(str(img_path))   # read image as numpy array
        if image is None:                   # skip if file isn't a valid image
            continue

        # Save original to processed folder
        cv2.imwrite(str(save_dir / img_path.name), image)

        # Generate N augmented copies
        for i in range(num_augmented):
            augmented = transform(image=image)["image"]  # apply random augmentations
            aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"  # e.g., pan_001_aug_3.jpg
            cv2.imwrite(str(save_dir / aug_name), augmented)  # save augmented image
