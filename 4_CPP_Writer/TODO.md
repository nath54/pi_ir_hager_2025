# Format
## To declare instance of a type, we need a string id. To access this object, the function needs to take this id as an argument


# Put all desired type and methods below.

Tensor init_tensor(dimensions: List[int])
'''
intialise un Tensor de dimension "dimensions"
'''

Tensor mult_tensor(X1: Tensor, X2: Tensor)
'''
Vérifie les dimensions et multiplie les Tensors "X1" et "X2" et retourne le résultat
'''

Tensor mult_tensor_scal(X1: Tensor, scal: int)
'''
multiplie le Tensor "X1" par le scalaire "scal"
'''

Tensor add_tensor(X1: Tensor, X2: Tensor)
'''
Vérifie les dimensions et additionne les Tensors "X1" et "X2" et retourne le résultat
'''

Tensor max_tensor_scal(X1: Tensor, scal: int)
'''
fait le max element-wise du tensor "X1" et du scalaire "scal" et retourne le résultat
'''

Tensor Softmax(X: Tensor, dim: int)
'''
renvoi le résultat du calcul suivant:
softmax(xi) = exp(xi) / Σ(exp(x))
le calcul est fait selon les lignes si "dim" = 1 et selon les colonnes si "dim" = 0
'''