import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

# Dataset folder
dataset_path = "dataset"

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Loop through each class folder
for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing: {folder}")

    for img_name in os.listdir(folder_path):

        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, img_name)

        try:
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            count = 0

            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=folder_path,
                save_prefix="aug",
                save_format="jpeg"
            ):
                count += 1

                # Create 25 augmented images per original image
                if count >= 25:
                    break

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("\nDataset Augmentation Completed Successfully!")