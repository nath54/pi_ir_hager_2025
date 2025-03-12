# Format for our code base
Everything is typed.


# Variables
Lower-case with underscore as separator
Examples :
- layer_nb
- global_counter

# Classes
Lower-case with capital letter, with capital letter as separators
Examples :
- ModelBlock
- Langage1Model

# Functions and methods
Lower-case with undescore as separator. Arguments: No space except after commas.
Examples :
- add_matrixes(matrix_a : Matrix, matrix_b : Matrix)

# Commentary
We are using python docstrings, with the following format :
    """
    Function's summary.

    Args:
        arg1 (type1): description
        arg2 (type2): description
        ...

    Returns:
    type: description