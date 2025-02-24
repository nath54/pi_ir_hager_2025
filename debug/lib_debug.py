#
from typing import Any


#
def debug_var0(var: Any, txt_sup: str = "") -> None:
    #
    n: int = 0
    #
    for attr in dir(var):
        if not attr.startswith("__") and attr not in ["denominator", "imag", "numerator", "real"]:
            n += 1
            debug_var( getattr(var, attr), txt_sup=f"{txt_sup}{'.' if txt_sup else ''}{attr}")
    #
    if isinstance(var, list):
        for i in range(len(var)):
            debug_var( var[i], txt_sup=f"{txt_sup}[{i}]")
    #
    if isinstance(var, dict):
        for key in var:
            debug_var( var[key], txt_sup=f"{txt_sup}['{key}']")
    #
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str) or isinstance(var, list) or isinstance(var, dict):
        print(f"DEBUG VAR | {txt_sup} = {var}")


#
def debug_var(var: Any, txt_sup: str = "") -> None:
    #
    n: int = 0
    #
    for attr in dir(var):
        #
        # if not attr.startswith("__") and attr not in ["denominator", "imag", "numerator", "real"]:
        #     n += 1
        #     debug_var( getattr(var, attr), txt_sup=f"{txt_sup}{'.' if txt_sup else ''}{attr}")
        #
        if attr in ["expr", "name", "func", "attr", "id", "value", "targets", "target", "annotation", "type_comment", "left", "right", "arg", "args", "op", "keywords", "slice", "dims", "elts", "operand", "generators"]:  # "body"
            n += 1
            debug_var( getattr(var, attr), txt_sup=f"{txt_sup}{'.' if txt_sup else ''}{attr}")
    #
    if isinstance(var, list):
        for i in range(len(var)):
            debug_var( var[i], txt_sup=f"{txt_sup}[{i}]")
    #
    if isinstance(var, dict):
        for key in var:
            debug_var( var[key], txt_sup=f"{txt_sup}['{key}']")
    #
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str) or isinstance(var, dict): # or isinstance(var, list):
        print(f"DEBUG VAR | {txt_sup} = {var}")
