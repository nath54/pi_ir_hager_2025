#
### Import Modules. ###
#
from typing import Any, Optional
#
import os
import inspect
import importlib


#
def list_classes(module: object) -> list[type]:
    """
    list all the classes defined inside a module.

    Args:
        module (object): The module to inspect.

    Returns:
        list[type]: A list of class objects found in the module.
    """

    #
    ### Initialize a list to store class objects, with type hint as list[type]. ###
    #
    classes: list[type] = [

        #
        ### m is a tuple of precise type :  tuple[str, type] , correponding to (name, value). ###
        #
        #
        ### Extract the second element of the tuple 'm', which is the class object itself. ###
        #
        m[1]

        #
        ### Iterate through members of the module that are classes, using inspect.getmembers with inspect.isclass filter. ###
        #
        for m in inspect.getmembers(module, inspect.isclass)
    ]

    #
    ### Return the list of class objects. ###
    #
    return classes


#
def import_module_from_filepath(filepath: str) -> object:
    """
    Imports a Python module from a filepath.

    Args:
        filepath (str): The path to the Python file to import as a module.

    Returns:
        object: The imported module object.
    """

    #
    ### Extract the module name from the filepath by removing the extension and path. ###
    #
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]

    #
    ### Create a module specification using the module name and filepath, spec is of type ModuleSpec. ###
    #
    spec: Optional[importlib.machinery.ModuleSpec] = importlib.util.spec_from_file_location(module_name, filepath)  # type: ignore

    #
    ### Check for errors. ###
    #
    if spec is None or spec.loader is None:  # type: ignore
        #
        raise ImportError(f"Error : can't load module from file : {filepath}")

    #
    ### Create a module object from the specification, module type is dynamically determined so using Any. ###
    #
    module: Any = importlib.util.module_from_spec(spec)  # type: ignore

    #
    ### Execute the module code in the module object's namespace, populating the module. ###
    #
    spec.loader.exec_module(module)  # type: ignore

    #
    ### Return the imported module object. ###
    #
    return module  # type: ignore

