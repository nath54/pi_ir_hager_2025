OUTPUT_PATH = "output.cpp"
"""
    Temporary, needs to be retrieved from the config file
"""


class CWriter:
    """
        A class containing all methods to write the second intermediate langage basic operations in C++.
    """
    
    def __init__(self, output_path : ):
        self.output_file = open(output_path)
        

cwriter = CWriter(OUTPUT_PATH)