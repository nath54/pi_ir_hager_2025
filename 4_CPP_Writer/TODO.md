# Format
## To declare instance of a type, we need a string id. To access this object, the function needs to take this id as an argument


# Put all desired type and methods below.

Tensor Softmax(X: Tensor, dim: int)
'''
renvoi le résultat du calcul suivant:
softmax(xi) = exp(xi) / Σ(exp(x))
le calcul est fait selon les lignes si dim = 1 et selon les colonnes si dim = 0
'''