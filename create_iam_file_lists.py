
import os

def create_file_list(base_dir, output_dir):
    """
    Generates train, validation, and test file lists from the IAM dataset structure.
    """
    sets = {
        "train": "train/images",
        "valid": "validation/images",
        "test": "test/images"
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for set_name, set_path in sets.items():
        image_dir = os.path.join(base_dir, set_path)
        output_filename = os.path.join(output_dir, f"list_{set_name}_iam.txt")

        if not os.path.isdir(image_dir):
            print(f"Warning: Directory not found, skipping: {image_dir}")
            continue

        try:
            # Get filenames and remove extensions
            filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

            with open(output_filename, 'w') as f:
                for name in sorted(filenames): # Sort for consistent order
                    f.write(name + '\n')
            
            print(f"Successfully generated: {output_filename} with {len(filenames)} entries.")

        except Exception as e:
            print(f"Error processing {image_dir}: {e}")

if __name__ == "__main__":
    iam_dataset_path = "datasets/iam_raw"
    lists_output_path = "Sets"
    
    print("Starting file list generation for IAM dataset...")
    create_file_list(iam_dataset_path, lists_output_path)
    print("\nGeneration complete.")
