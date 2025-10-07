import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf

from model_to_export import Model

# 1. Charger le modèle ONNX
onnx_model = onnx.load('model.onnx')

# 2. Définir les noms des entrées du modèle ONNX (essentiel)
# L'exemple ci-dessous utilise 'input_name' et suppose une forme (batch_size, channels, height, width)
input_names = ['input_name'] # Remplacez par le nom réel de l'entrée de votre modèle ONNX

# 3. Convertir en modèle Keras
k_model = onnx_to_keras(onnx_model, input_names)

# 4. Sauvegarder le modèle Keras au format .keras (Keras 3)
k_model.save('converted_model.keras')