import torch
import onnx
from onnx2keras import onnx_to_keras
from keras.models import save_model  # ou simplement utiliser k_model.save()

# 1. Définir et charger le modèle PyTorch
model = MyModelClass([1,6,38,5])  # Classe du modèle à adapter
model.load_state_dict(torch.load('modele.pth'))
model.eval()

# 2. Exporter en ONNX (il faut fournir un exemple d’entrée de taille correcte)
dummy_input = torch.randn(1,6,38,5)  # adapter (C,H,W) à l’entrée du modèle
torch.onnx.export(model, dummy_input, 'modele.onnx',
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

# 3. Charger le modèle ONNX et convertir en Keras
onnx_model = onnx.load('modele.onnx')
k_model = onnx_to_keras(onnx_model, ['input'])

# 4. Sauvegarder le modèle Keras au format .keras
k_model.save('modele.keras')
