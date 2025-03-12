OUTPUT_PATH = "output.cpp"
"""
    Temporary, needs to be retrieved from the config file
"""


class CWriter:
    """
        A class containing all methods to write the second intermediate langage basic operations in C++.
    """
    
    def __init__(self, output_path : str):
        self.output_file = open(output_path,'w')
    
    def write(self, to_write : str):
        """
            Write to the output file.
        Args:
            to_write (str): the string to write to the output file.
        """
        self.output_file.write(to_write)
    
    
        
    def __del__(self):
        self.output_file.close()

cwriter = CWriter(OUTPUT_PATH)