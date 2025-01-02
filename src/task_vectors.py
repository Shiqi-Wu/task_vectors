import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """
        Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.

        Args:
            pretrained_checkpoint (str, optional): Path to the pretrained model checkpoint.
            finetuned_checkpoint (str, optional): Path to the finetuned model checkpoint.
            vector (dict, optional): Precomputed task vector (if already available).
        """
        if vector is not None:
            # If a precomputed task vector is provided, directly use it.
            self.vector = vector
        else:
            # Ensure both pretrained and finetuned checkpoints are provided if vector is not.
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            
            with torch.no_grad():  # Disable gradient calculations for efficiency.
                # Load state dictionaries from the provided checkpoint files.
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                
                self.vector = {}
                # Compute the task vector as the difference between finetuned and pretrained weights.
                for key in pretrained_state_dict:
                    # Skip keys with integer or binary types since they don't require updates.
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    # Compute the difference for all eligible keys and store in the vector.
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """
        Add two task vectors together.

        Args:
            other (TaskVector): Another TaskVector to add.

        Returns:
            TaskVector: A new TaskVector with summed values.
        """
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                # Ensure the key exists in both task vectors before adding.
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                # Sum the corresponding entries for the key.
                new_vector[key] = self.vector[key] + other.vector[key]
        # Return a new TaskVector instance containing the summed vector.
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        """
        Allow addition with other types or using the built-in sum() function.

        Args:
            other (TaskVector or int): The other object to add (int is treated as 0).

        Returns:
            TaskVector: The sum of the vectors.
        """
        if other is None or isinstance(other, int):
            # If the other operand is None or an integer, treat it as neutral (0).
            return self
        # Otherwise, perform a standard addition.
        return self.__add__(other)

    def __neg__(self):
        """
        Negate a task vector (multiply all values by -1).

        Returns:
            TaskVector: A new TaskVector with negated values.
        """
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                # Negate each entry in the task vector.
                new_vector[key] = - self.vector[key]
        # Return a new TaskVector instance containing the negated vector.
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """
        Apply a task vector to a pretrained model to create a modified model.

        Args:
            pretrained_checkpoint (str): Path to the pretrained model checkpoint.
            scaling_coef (float, optional): Scaling coefficient for the task vector. Defaults to 1.0.

        Returns:
            torch.nn.Module: The modified model with the task vector applied.
        """
        with torch.no_grad():
            # Load the pretrained model from the checkpoint.
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            # Get the state dictionary of the pretrained model.
            pretrained_state_dict = pretrained_model.state_dict()
            
            for key in pretrained_state_dict:
                # Warn if a key in the pretrained model does not exist in the task vector.
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                # Update the model weights by adding the scaled task vector to the pretrained weights.
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        
        # Load the updated state dictionary into the model (allowing missing keys).
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        # Return the modified model.
        return pretrained_model