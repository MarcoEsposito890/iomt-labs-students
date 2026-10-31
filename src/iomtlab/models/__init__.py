"""The course models and their cost profiles.

Lab 4 trains SimpleCNN, ExampleCNN (the example network of the theory),
BottleneckCNN (the same with a bottleneck after pool2) and DeepConvLSTM;
ExampleCNN is the model of Labs 5 to 8. DeepCNN and EarlyExitCNN are kept
for the tests and the theory. The cost
functions count parameters, MACs and bytes (Labs 4, 5, 6, 8).
"""
from iomtlab.models.simple_cnn import SimpleCNN
from iomtlab.models.deep import DeepCNN, DeepConvLSTM, EarlyExitCNN
from iomtlab.models.example_cnn import ExampleCNN, BottleneckCNN
from iomtlab.models.cost import (
    LayerCost, ModelCost, layer_profile, count_parameters,
    model_size_bytes, federated_round_bytes, module_macs, counted_modules, uncounted_modules,
)

__all__ = [
    "SimpleCNN", "DeepCNN", "DeepConvLSTM", "EarlyExitCNN", "ExampleCNN", "BottleneckCNN",
    "LayerCost", "ModelCost", "layer_profile", "count_parameters",
    "model_size_bytes", "federated_round_bytes", "module_macs", "counted_modules",
    "uncounted_modules",
]
from iomtlab.models.gap_cnn import GapCNN, GapCNNFine   # noqa: E402,F401  soluzioni dei Lab 4 e 6
