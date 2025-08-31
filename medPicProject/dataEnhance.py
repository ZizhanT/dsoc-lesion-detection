import os
from glob import glob
from PIL import Image
import torch
import shutil
from torchvision import transforms as T
from ultralytics import YOLO

# Define the augmentation transformations
def get_augmentation_transform(image_size=640, augment_horizontal_flip=True):
    return T.Compose([
        T.Resize((image_size, image_size)),  # Resize image to the specified size
        T.RandomHorizontalFlip() if augment_horizontal_flip else T.Lambda(lambda x: x),  # Random horizontal flip
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
        T.RandomRotation(degrees=90),  # Random rotation
        T.ToTensor(),  # Convert image to tensor

    ])

# Convert tensor to PIL Image
def tensor_to_pil(tensor):
    # Convert the tensor to a PIL image (expects tensor to be in [0, 1] range)
    tensor = tensor.permute(1, 2, 0)  # Change shape from C x H x W to H x W x C
    tensor = tensor * 255  # Convert range from [0, 1] to [0, 255]
    tensor = tensor.to(torch.uint8)  # Convert to uint8 type
    return Image.fromarray(tensor.numpy())  # Convert the tensor to a PIL Image

def augment_and_save_images(image_dir, label_dir, output_image_dir, output_label_dir, transform):
    """
    Apply transformations to the images and save the augmented images and labels into new directories.
    :param image_dir: Path to the original image directory
    :param label_dir: Path to the original label directory
    :param output_image_dir: Path to save augmented images
    :param output_label_dir: Path to save labels
    :param transform: The transformation to apply
    """
    # Create output directories if they do not exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get the image and label files
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))  # Assuming images are .jpg, adjust if needed
    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))  # Assuming labels are .txt, adjust if needed

    for image_file, label_file in zip(image_files, label_files):
        # Load image
        image = Image.open(image_file).convert("RGB")

        # Apply transformations
        transformed_image = transform(image)

        # Convert tensor back to PIL image before saving
        pil_image = tensor_to_pil(transformed_image)

        # Save the augmented image
        augmented_image_path = os.path.join(output_image_dir, os.path.basename(image_file))
        pil_image.save(augmented_image_path)

        # Copy the corresponding label file to the new label folder
        augmented_label_path = os.path.join(output_label_dir, os.path.basename(label_file))
        shutil.copy(label_file, augmented_label_path)



# Training function using YOLO and custom data augmentation
def train():
    # Define directories
    train_image_dir = "E:/medPicProject/img/train/images"  # Original training images
    train_label_dir = "E:/medPicProject/img/train/labels"  # Original training labels
    valid_image_dir = "E:/medPicProject/img/valid/images"  # Original validation images
    valid_label_dir = "E:/medPicProject/img/valid/labels"  # Original validation labels

    # Define output directories for augmented data
    train_enhanced_image_dir = "E:/medPicProject/img/train_enhanced/images"
    train_enhanced_label_dir = "E:/medPicProject/img/train_enhanced/labels"
    valid_enhanced_image_dir = "E:/medPicProject/img/valid_enhanced/images"
    valid_enhanced_label_dir = "E:/medPicProject/img/valid_enhanced/labels"

    #   Define the transform (augmentation)
    transform = get_augmentation_transform(image_size=640, augment_horizontal_flip=True)

    # Apply transformations and save the augmented data
    augment_and_save_images(train_image_dir, train_label_dir, train_enhanced_image_dir, train_enhanced_label_dir, transform)
    augment_and_save_images(valid_image_dir, valid_label_dir, valid_enhanced_image_dir, valid_enhanced_label_dir, transform)

    print("Data augmentation completed and saved.")

    # Load the pre-trained YOLO model
    model = YOLO("yolo11n.pt")  # Pre-trained model file

    # Train the model using the YAML file
    model.train(
        data="enhanced.yaml",  # Path to your YAML dataset configuration file
        epochs=2000,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="cuda:0",  # Using GPU (if available)
    )


if __name__ == '__main__':
    train()
