"""
    An object used to translate the intermediate langage and write the result in a .cpp file.
    To import, use
    from c_writer import c_writer
    Then, call methods directly on the cwriter object
"""
OUTPUT_PATH = "output.cpp"
"""
    Temporary, needs to be retrieved from the config file
"""


class CWriter:
    """
        A class containing all methods to write the second intermediate langage basic operations in C++.
    """
    
    def __init__(self, output_path: str):
        self.output_file = open(output_path,'w')
    
    def write(self, to_write: str) -> str:
        """
            Write to the output file.
        Args:
            to_write (str): the string to write to the output file.
        Returns:
            str: the string written.
        """
        self.output_file.write(to_write)
        
    def __del__(self):
        self.output_file.close()
    

c_writer = CWriter(OUTPUT_PATH)