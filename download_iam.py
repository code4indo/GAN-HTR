
import os
from datasets import load_dataset
from PIL import Image

def download_and_prepare_iam(base_path="datasets/iam_raw"):
    """
    Downloads the Teklia/IAM-line dataset from Hugging Face and 
    organizes it into train, validation, and test sets in a specified directory.

    Each image is saved, and its corresponding transcription is stored in a 
    lines.txt file within the respective set's folder.
    """
    # 1. Load the dataset from Hugging Face
    print("Loading Teklia/IAM-line dataset from Hugging Face...")
    dataset = load_dataset("Teklia/IAM-line")
    print("Dataset loaded successfully.")

    # 2. Create directories and process each split
    for split in ["train", "validation", "test"]:
        print(f"Processing '{split}' split...")
        
        # Define paths
        image_dir = os.path.join(base_path, split, "images")
        labels_file = os.path.join(base_path, split, "lines.txt")

        # Create directories if they don't exist
        os.makedirs(image_dir, exist_ok=True)

        with open(labels_file, "w", encoding="utf-8") as f_labels:
            # Iterate over the dataset split
            for i, item in enumerate(dataset[split]):
                image = item["image"]
                text = item["text"]
                image_name = f"{split}_{i:04d}.png"
                image_path = os.path.join(image_dir, image_name)

                # Save the image
                image.save(image_path)

                # Write the mapping of image name to text
                f_labels.write(f"{image_name}\t{text}\n")
        
        print(f"Finished processing '{split}' split. Images are in '{image_dir}', labels in '{labels_file}'.")

if __name__ == "__main__":
    download_and_prepare_iam()
    print("\nDataset preparation complete.")
