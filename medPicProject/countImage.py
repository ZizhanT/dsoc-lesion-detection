import yaml
import os
from glob import glob
from ultralytics import YOLO

def count_images_in_directory(directory):
    #print(f"Checking directory: {directory}")  # Debug print
    image_files = glob(os.path.join(directory, "*.jpg"))  # Adjust extension if necessary
    #print(f"Found {len(image_files)} images in {directory}")  # Debug print
    return len(image_files)

def print_image_counts_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # Ensure the paths are correct
    base_path = data['path']
    train_image_dir = os.path.join(base_path, data['train'])  # Full path to the training images
    val_image_dir = os.path.join(base_path, data['val'])  # Full path to the validation images

    # Count the images in the directories
    train_image_count = count_images_in_directory(train_image_dir)
    val_image_count = count_images_in_directory(val_image_dir)

    print(f"Training set contains {train_image_count} images.")
    print(f"Validation set contains {val_image_count} images.")
    #print(f"Train directory path: {train_image_dir}")
    #print(f"Validation directory path: {val_image_dir}")

def main():
    print_image_counts_from_yaml("yoloData.yaml")
    print_image_counts_from_yaml("shuffledData.yaml")
    print_image_counts_from_yaml("enhanced.yaml") # Path to your YAML dataset configuration file
if __name__ == '__main__':
    main()