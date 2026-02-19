import os
import cv2

# Dataset Path
DATASET_PATH = "dataset"

# Target Image Size
IMG_SIZE = 224

# -------------------------------
# Function: Preprocess Images
# -------------------------------
def preprocess_folder(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # Read Image
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convert to Grayscale (recommended)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save back
        cv2.imwrite(img_path, img)


# -------------------------------
# Apply Preprocessing to Dataset
# -------------------------------
print("Starting Manual Preprocessing...")

for split in ["train", "valid", "test"]:
    split_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(split_path):
        continue

    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)

        print(f"Processing: {split}/{class_name}")
        preprocess_folder(class_folder)

print("\nManual Preprocessing Completed Successfully!")
