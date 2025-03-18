# Format
## To declare instance of a type, we need a string id. To access this object, the function needs to take this id as an argument


# Put all desired type and methods below.

Tensor init_tensor(dimensions: List[int])
'''
intialise un Tensor de dimension "dimensions"
'''

Tensor mult_tens(X1: Tensor, X2: Tensor)
'''
Vérifie les dimensions et multiplie les Tensors "X1" et "X2" et retourne le résultat
'''

Tensor mult_tens_scal(X1: Tensor, scal: int)
'''
multiplie le Tensor "X1" par le scalaire "scal"
'''

Tensor add_tens(X1: Tensor, X2: Tensor)
'''
Vérifie les dimensions et additionne les Tensors "X1" et "X2" et retourne le résultat
'''

Tensor max_tens_scal(X1: Tensor, scal: int)
'''
fait le max element-wise du tensor "X1" et du scalaire "scal" et retourne le résultat
'''

int moy_tens(X1: Tensor)
'''
retourne la moyenne des éléments du tensor "X1"
'''

int var_tens(X1: Tensor)
'''
retrourne la variance des éléments du tensor "X1"
'''

int sqrt(scal: int)
'''
fait la racine du scalaire "scal"
'''

int add_tens_scal(X1: Tensor, scal: int)
'''
fait l'addition element-wise du tensor "X1" et du scalaire "scal"
'''

Tensor Softmax(X: Tensor, dim: int)
'''
renvoi le résultat du calcul suivant:
softmax(xi) = exp(xi) / Σ(exp(x))
le calcul est fait selon les lignes si "dim" = 1 et selon les colonnes si "dim" = 0
'''