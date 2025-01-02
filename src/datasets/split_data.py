import os
import shutil
import random

def split_dataset_by_label(images_dir, output_dir, train_ratio=0.8):
    """
    Splits an image dataset organized by labels into train and test sets while preserving the class folder structure.

    Args:
        images_dir (str): Path to the original dataset folder (images are organized into class folders).
        output_dir (str): Path to the output dataset folder (train and test subdirectories will be created).
        train_ratio (float): The ratio of images used for training (default: 80% for training).
    """
    # Iterate over each class folder (label folder) in the dataset
    for label in os.listdir(images_dir):
        label_dir = os.path.join(images_dir, label)
        
        # Skip non-folder items
        if not os.path.isdir(label_dir):
            continue
        
        # Get all images in the current label directory
        all_images = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        random.shuffle(all_images)  # Shuffle the images to ensure random split

        # Calculate the split point between training and testing
        train_count = int(len(all_images) * train_ratio)
        train_images = all_images[:train_count]  # First `train_ratio` percent for training
        test_images = all_images[train_count:]  # Remaining for testing

        # Create output directories for train and test splits
        train_label_dir = os.path.join(output_dir, "train", label)
        test_label_dir = os.path.join(output_dir, "val", label)
        os.makedirs(train_label_dir, exist_ok=True)  # Create train directory if it doesn't exist
        os.makedirs(test_label_dir, exist_ok=True)  # Create test directory if it doesn't exist

        # Copy the training images to the corresponding train directory
        for img in train_images:
            shutil.copy(os.path.join(label_dir, img), os.path.join(train_label_dir, img))
        
        # Copy the testing images to the corresponding test directory
        for img in test_images:
            shutil.copy(os.path.join(label_dir, img), os.path.join(test_label_dir, img))
    
    # Print completion message
    print("Dataset split completed!")
    print(f"Training set directory: {os.path.join(output_dir, 'train')}")
    print(f"Testing set directory: {os.path.join(output_dir, 'val')}")

# Example usage
split_dataset_by_label('/home/shiqi/code/task_vectors/data/EuroSAT_RGB', '/home/shiqi/code/task_vectors/experimental_results/data/EuroSAT_splits', train_ratio=0.8)