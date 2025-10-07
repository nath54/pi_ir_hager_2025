import torch
import torch.nn as nn
from model_to_export import Model
# 1. Charger votre modèle PyTorch (exemple)
model: nn.Module = Model()
model.load_state_dict(torch.load("./Tests_Concurrents/test_keras/model_to_export_weights.pth", weights_only=True))
model.eval()

# 2. Définir une entrée factice (remplacez les dimensions par celles de votre modèle)
dummy_input = torch.randn(1, 6, 38, 5) 

# 3. Exporter vers ONNX
torch.onnx.export(model,
                  dummy_input,
                  "model.onnx",
                  opset_version=18, # Tenter une version plus récente
                  input_names=['input'],
                  output_names=['output'],
                  dynamo=True)