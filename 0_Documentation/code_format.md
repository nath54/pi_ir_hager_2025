# Format for our code base

Everything is typed.

## Variables

### Variable's definition

Names lower-case with underscore as separator, type with spaces
Examples :

```py
#
example_layer : Layer = Layer()

#
global_counter : int = 0
```

## Classes

### Classe's definition

Names are ower-case with capital letter, with capital letter as separators
Examples :

```py
#
class ModelBlock():

    # ...

#
class Langage1Model():

    # ...

```

### Classe's docstring

Documentation with python docstrings, with the following format :
```py

#
class MyClass():
    """
    Class's summary

    Attributes:
        attr1 (type1): description
        attr2 (type2): description
        ...
<<<<<<< HEAD
=======

    """

```

>>>>>>> 1caf8b8cbf0c7b0cb2691d2a38477a68e6f339e7
The class's methods have their own documentation

## Functions and methods

### Function's definition

Lower-case with undescore as separator. Arguments: Spaces after commas. Return is typed.
Examples :

```py

#
def add_matrixes(matrix_a : Matrix, matrix_b : Matrix) -> return_type:

    # ...

```

### Function's docstring

Documentation with python docstrings, with the following format :


```py

def my_function(arg1: type1, arg2: type2, ...) -> type_out:
    """
    Function's summary.

    Args:
        arg1 (type1): description
        arg2 (type2): description
        ...

    Returns:
        type_out: description

    """

    # ...

```

## Modules (eg. each python script file)

A docstring at describing the intended way of using and importing the module. Don't describe all functions and variables.

The docstring of a module is before all the imports and other stuffs.
