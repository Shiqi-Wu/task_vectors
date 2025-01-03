import torch
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

project_root = os.path.abspath(os.getcwd())
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

class TaskVector2:
    def __init__(self, pretrained_state_dict=None, finetuned_state_dict=None, vector=None):
        """
        Initializes the task vector.

        Args:
            pretrained_state_dict (str): Path to the state_dict of the pretrained model.
            finetuned_state_dict (str): Path to the state_dict of the fine-tuned model.
            vector (dict): Precomputed task vector (optional).
        """
        if vector is not None:
            # Use the precomputed task vector if provided
            self.vector = vector
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None
            with torch.no_grad():
                # Load state_dict files
                pretrained_weights = torch.load(pretrained_state_dict, map_location="cpu")
                finetuned_weights = torch.load(finetuned_state_dict, map_location="cpu")

                # Compute the task vector
                self.vector = {
                    key: finetuned_weights[key] - pretrained_weights[key]
                    for key in pretrained_weights
                    if pretrained_weights[key].dtype not in [torch.int64, torch.uint8]
                }

    def dot(self, other):
        """
        Computes the dot product between two task vectors.

        Args:
            other (TaskVector2): Another task vector.

        Returns:
            float: The dot product result.
        """
        return sum(torch.sum(self.vector[key] * other.vector[key]) for key in self.vector if key in other.vector)

    def norm(self):
        """
        Computes the L2 norm of the task vector.

        Returns:
            float: The norm result.
        """
        return math.sqrt(sum(torch.sum(self.vector[key]**2) for key in self.vector))

    def cosine_similarity(self, other):
        """
        Computes the cosine similarity between two task vectors.

        Args:
            other (TaskVector2): Another task vector.

        Returns:
            float: The cosine similarity.
        """
        return self.dot(other) / (self.norm() * other.norm())

    @staticmethod
    def convert_to_state_dict(input_checkpoint, output_checkpoint):
        """
        Converts a full model checkpoint to a state_dict file.

        Args:
            input_checkpoint (str): Path to the full model checkpoint.
            output_checkpoint (str): Path to save the converted state_dict file.
        """
        # Load the full model checkpoint
        full_model = torch.load(input_checkpoint, map_location="cpu")
        
        # Extract state_dict from the full model
        if hasattr(full_model, "state_dict"):
            state_dict = full_model.state_dict()
        else:
            state_dict = full_model  # If already in state_dict format
        
        # Save the state_dict to the specified output file
        torch.save(state_dict, output_checkpoint)
        print(f"State dict saved to {output_checkpoint}")

def plot_task_vector_angles(base_path, model_name, datasets, save_path=None):
    """
    Compute and save a heatmap of angles between task vectors.

    Args:
        base_path (str): The root directory containing checkpoints.
        model_name (str): The model name (e.g., "ViT-B-16").
        datasets (list): List of dataset names.
        save_path (str, optional): If provided, save the heatmap to the specified path.
    """
    model_path = os.path.join(base_path, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    task_vectors = {}
    for dataset in datasets:
        finetuned_full_path = os.path.join(model_path, dataset, "finetuned.pt")
        zeroshot_full_path = os.path.join(model_path, dataset, "zeroshot.pt")
        finetuned_state_dict_path = os.path.join(model_path, dataset, "finetuned_state_dict.pt")
        zeroshot_state_dict_path = os.path.join(model_path, dataset, "zeroshot_state_dict.pt")

        # Convert full model to state_dict if necessary
        if not os.path.exists(finetuned_state_dict_path):
            TaskVector2.convert_to_state_dict(finetuned_full_path, finetuned_state_dict_path)
        if not os.path.exists(zeroshot_state_dict_path):
            TaskVector2.convert_to_state_dict(zeroshot_full_path, zeroshot_state_dict_path)

        # Initialize task vector
        task_vectors[dataset] = TaskVector2(pretrained_state_dict=zeroshot_state_dict_path, 
                                            finetuned_state_dict=finetuned_state_dict_path)

    num_datasets = len(datasets)
    angle_matrix = np.zeros((num_datasets, num_datasets))

    for i, dataset1 in enumerate(datasets):
        for j, dataset2 in enumerate(datasets):
            if i != j and dataset1 in task_vectors and dataset2 in task_vectors:
                similarity = task_vectors[dataset1].cosine_similarity(task_vectors[dataset2])
                angle = math.acos(similarity) * 180 / math.pi  # Convert to degrees
                angle_matrix[i, j] = angle

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    cax = plt.imshow(angle_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(cax, label="Angle (Degrees)")

    plt.xticks(range(num_datasets), datasets, rotation=45, ha="right")
    plt.yticks(range(num_datasets), datasets)

    for i in range(num_datasets):
        for j in range(num_datasets):
            plt.text(j, i, f"{angle_matrix[i, j]:.1f}", ha="center", va="center", color="black")

    plt.title(f"Angle Between Task Vectors for {model_name} (Heatmap)")
    plt.tight_layout()

    # Save the heatmap
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")

def plot_task_vector_ratios(base_path, model_name, datasets, save_path=None):
    """
    Compute and save a heatmap of projection norm ratios between task vectors.

    Args:
        base_path (str): The root directory containing checkpoints.
        model_name (str): The model name (e.g., "ViT-B-16").
        datasets (list): List of dataset names.
        save_path (str, optional): If provided, save the heatmap to the specified path.
    """
    model_path = os.path.join(base_path, model_name)

    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    task_vectors = {}
    for dataset in datasets:
        finetuned_full_path = os.path.join(model_path, dataset, "finetuned.pt")
        zeroshot_full_path = os.path.join(model_path, dataset, "zeroshot.pt")
        finetuned_state_dict_path = os.path.join(model_path, dataset, "finetuned_state_dict.pt")
        zeroshot_state_dict_path = os.path.join(model_path, dataset, "zeroshot_state_dict.pt")

        # Convert full model to state_dict if necessary
        if not os.path.exists(finetuned_state_dict_path):
            TaskVector2.convert_to_state_dict(finetuned_full_path, finetuned_state_dict_path)
        if not os.path.exists(zeroshot_state_dict_path):
            TaskVector2.convert_to_state_dict(zeroshot_full_path, zeroshot_state_dict_path)

        # Initialize task vector
        task_vectors[dataset] = TaskVector2(pretrained_state_dict=zeroshot_state_dict_path, 
                                            finetuned_state_dict=finetuned_state_dict_path)

    num_datasets = len(datasets)
    ratio_matrix = np.zeros((num_datasets, num_datasets))

    for i, dataset1 in enumerate(datasets):
        for j, dataset2 in enumerate(datasets):
            if dataset1 in task_vectors and dataset2 in task_vectors:
                task_vector1 = task_vectors[dataset1]
                task_vector2 = task_vectors[dataset2]

                # Compute projection norm ratio
                projection_norm = task_vector1.dot(task_vector2) / task_vector2.norm()
                ratio = projection_norm / task_vector2.norm()
                ratio_matrix[i, j] = ratio

    # Plot the heatmap for ratios
    plt.figure(figsize=(10, 8))
    cax = plt.imshow(ratio_matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(cax, label="Projection Norm Ratio")

    # Add axis labels for datasets
    plt.xticks(range(num_datasets), datasets, rotation=45, ha="right")
    plt.yticks(range(num_datasets), datasets)

    # Annotate the heatmap with ratio values
    for i in range(num_datasets):
        for j in range(num_datasets):
            plt.text(j, i, f"{ratio_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    # Add title indicating projection relationship
    plt.title(f"Projection Norm Ratios: {model_name} Task Vectors", fontsize=14)
    plt.xlabel("Dataset2 (Target of Projection)")
    plt.ylabel("Dataset1 (Projected Task Vector)")
    plt.tight_layout()

    # Save the heatmap
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        output_path = save_path.replace(".png", "_ratios.png")
        plt.savefig(output_path)
        print(f"Heatmap saved to {output_path}")


# Main script
if __name__ == "__main__":
    base_path = os.path.join(project_root, "checkpoints")
    models = ["ViT-B-16", "ViT-B-32"]
    datasets = ["DTDVal", "EuroSATVal", "GTSRBVal", "MNISTVal", "SVHNVal"]

    for model in models:
        save_file = f"/home/shiqi/code/task_vectors/experimental_results/results/{model}_heatmap.png"
        plot_task_vector_angles(base_path, model, datasets, save_path=save_file)
        save_ratio_file = f"/home/shiqi/code/task_vectors/experimental_results/results/{model}_heatmap_ratios.png"
        plot_task_vector_ratios(base_path, model, datasets, save_path=save_ratio_file)