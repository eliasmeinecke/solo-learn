from itertools import product
from typing import List, Dict, Any, Union, Tuple, Optional

import torch
import torch.nn as nn


def parameter_iterator(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from a dictionary of lists.
    """
    keys, values = [], []
    for k, v in params.items():
        if v is not None:
            keys.append(k)
            values.append(v)

    return [dict(zip(keys, combination)) for combination in product(*values)]


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
            self,
            out_dim: int,
            num_classes: int,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        if x.shape[-1] != self.out_dim:
            raise ValueError(f"Output shape {x.shape[-1]} does not match the expected input dimension {self.out_dim}.")
        return self.linear(x)


def create_linear_input_cnn(x: Dict[str, torch.Tensor], layer_name: str, pool: str = "avg") -> Tuple[
    bool, torch.Tensor]:
    """
    Create the input for the linear head from the output of the CNN.

    Args:
        x: The output of the CNN.
        layer_name: The name of the layer to use.
        pool: The pooling method to use. Can be "avg" or "flatten".
    """
    if layer_name not in x:
        print(f"Layer {layer_name} not found in output.")
        return False, None

    # use the specified layer
    intermediate_output = x[layer_name]  # (batch, dim, height, width)

    if pool == "avg":
        output = torch.mean(intermediate_output, dim=(2, 3))
    elif pool == "flatten":
        output = intermediate_output.view(intermediate_output.shape[0], -1)
    else:
        raise ValueError(f"Pool {pool} is not supported.")

    return True, output.contiguous()


class CNNLinearClassifier(LinearClassifier):
    def __init__(
            self,
            out_dim: int,
            num_classes: int,
            layer_name: str,
            pool: str = "avg",
    ):
        super().__init__(out_dim, num_classes)
        self.layer_name = layer_name
        self.pool = pool

    def forward(self, x):
        ret, output = create_linear_input_cnn(x, layer_name=self.layer_name, pool=self.pool)
        if not ret:
            raise RuntimeError(f"Linear layer returned no output.")
        return super().forward(output)


class AllClassifiers(nn.Module):
    """
    A wrapper for multiple classifiers.
    """

    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def scale_lr(learning_rates: float, batch_size: int, devices: int) -> float:
    return learning_rates * (batch_size * devices) / 256.0


def setup_linear_classifiers(
        sample_output: torch.Tensor,
        learning_rates: List[float],
        batch_size: int,
        devices: int,
        num_classes: int,
        layer_names: Optional[List[str]] = None,
        pool: Optional[str] = None

) -> Union[AllClassifiers, List[Dict[str, Any]]]:
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for _lr in sorted(learning_rates):
        lr = scale_lr(_lr, batch_size, devices)

        if layer_names is None:
            out_dim = sample_output.shape[-1]
            linear_classifier = LinearClassifier(out_dim, num_classes=num_classes)
            clf_str = f"classifier-lr_{lr:.8f}".replace(".", ":")
            linear_classifiers_dict[clf_str] = linear_classifier
            optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

        else:
            for layer_name in layer_names:
                out_dim = sample_output[layer_name].shape[1]
                linear_classifier = CNNLinearClassifier(out_dim, num_classes=num_classes, layer_name=layer_name,
                                                        pool=pool)
                clf_str = f"classifier-layer={layer_name}-pool={pool}-lr_{lr:.8f}".replace(".", ":")
                linear_classifiers_dict[clf_str] = linear_classifier

                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups
