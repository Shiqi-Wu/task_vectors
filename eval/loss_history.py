import torch
import matplotlib.pyplot as plt
import os

# Define a function to load and plot loss history
def plot_loss_history(folder_path, model_name, save_path):
    """
    Plot loss history from .pt files in the specified folder.
    
    Parameters:
    - folder_path (str): Path to the directory containing the loss_history.pt files.
    - model_name (str): Name of the model (e.g., "ViT-B-16").
    """
    # Create a new plot
    plt.figure(figsize=(10, 6))
    datasets = os.listdir(folder_path)
    for dataset in datasets:
        dataset_path = os.path.join(folder_path, dataset, "loss_history.pt")
        if os.path.isfile(dataset_path):
            # Load the loss history file
            loss_history = torch.load(dataset_path)
            
            # Plot the loss history
            plt.plot(loss_history, label=dataset)
    
    # Configure and display the plot
    plt.title(f"Loss History for {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"{model_name}_loss_history.png"))
    plt.close()
    
# Example usage
project_root = os.path.abspath(os.getcwd())
base_path = os.path.join(project_root, "checkpoints")
save_path = os.path.join(project_root, "experimental_results/results")
model_folders = ["ViT-B-16", "ViT-B-32"]

for model in model_folders:
    model_path = os.path.join(base_path, model)
    print(f"Model: {model_path}")
    if os.path.exists(model_path):
        plot_loss_history(model_path, model, save_path)
        