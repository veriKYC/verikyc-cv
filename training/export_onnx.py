import torch
import torch.nn as nn
from torchvision import models

# Rebuild model architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)

# Load trained weights
model.load_state_dict(torch.load("app/models/efficientnet_b0.pth"))
model.eval()

# Dummy input — batch of 1, 3 channels, 224x224
dummy = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(model, dummy, "app/models/efficientnet_b0.onnx",
                    input_names=["image"],
                    output_names=["prediction"])

print("ONNX model exported")