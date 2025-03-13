# Format for our code base
Everything is typed.


# Variables
Names lower-case with underscore as separator, type with spaces
Examples :
- example_layer : Layer
- global_counter : int

# Classes
Names are ower-case with capital letter, with capital letter as separators
Examples :
- ModelBlock
- Langage1Model

Documentation with python docstrings, with the following format :
    """
    Class's summary.

    Attributes:
        attr1 (type1): description
        attr2 (type2): description
        ...

The class's methods have their own documentation

# Functions and methods
Lower-case with undescore as separator. Arguments: Spaces after commas. Return is typed.
Examples :
- def add_matrixes(matrix_a : Matrix, matrix_b : Matrix) -> return_type:

Documentation with python docstrings, with the following format :
    """
    Function's summary.

    Args:
        arg1 (type1): description
        arg2 (type2): description
        ...

    Returns:
    type: description

# Modules
A docstring at describing the intended way of using and importing the module. Don't describe all functions and variables.