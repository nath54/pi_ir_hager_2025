#
### Import modules. ###
#
from typing import Optional, Any, cast
#
import ast
import argparse


#
def display_ast_tree(node: ast.AST, indent: int=0, last_is_list_item: bool=False, parent_is_list: bool=False, continuation_columns: Optional[set[int]]=None):
    """
    Recursively displays the AST tree with formatted output.
    """

    #
    if continuation_columns is None:
        #
        continuation_columns = set()

    #
    ### Build the prefix with continuation lines. ###
    #
    prefix: str = ""
    #
    i: int
    #
    for i in range(0, indent, 2):
        #
        if i in continuation_columns:
            #
            prefix += "│ "
        #
        else:
            #
            prefix += "  "

    #
    ### Determine the branch character for the current node. ###
    #
    if parent_is_list:
        #
        if last_is_list_item:
            #
            prefix += "└─ "
        #
        else:
            #
            prefix += "├─ "

    #
    ### Get node name. ###
    #
    node_name = type(node).__name__

    #
    ### Display the current node. ###
    #
    print(f"{prefix}\033[1;32m{node_name}\033[m")

    #
    ### Filter out attributes that are not fields of the AST node. ###
    #
    ast_fields = [field for field in node._fields]

    #
    ### Update continuation columns for child attributes. ###
    #
    new_continuation_columns: set[int] = continuation_columns.copy()
    #
    if parent_is_list and not last_is_list_item:
        #
        new_continuation_columns.add(indent)

    #
    ### Iterate through each field of the node to display attributes and child nodes. ###
    #
    for i, field in enumerate(ast_fields):

        #
        attr_value: Optional[Any] = getattr(node, field, None)
        attr_type: str = type(attr_value).__name__
        is_last_field: bool = (i == (len(ast_fields) - 1))

        #
        ### Build attribute prefix with continuation lines. ###
        #
        attr_prefix: str = ""
        attr_indent: int = indent + 2 if parent_is_list else indent + 2

        #
        j: int
        #
        for j in range(0, attr_indent, 2):
            #
            if j in new_continuation_columns:
                #
                attr_prefix += "│ "
            #
            else:
                #
                attr_prefix += "  "

        #
        ### Add branch character for attribute. ###
        #
        if is_last_field:
            #
            attr_prefix += "└─"
        #
        else:
            #
            attr_prefix += "├─"

        #
        ### Update continuation columns for attribute children. ###
        #
        attr_continuation_columns = new_continuation_columns.copy()
        #
        if not is_last_field:
            #
            attr_continuation_columns.add(attr_indent)

        #
        ### Handle different types of attribute values. ###
        #
        if isinstance(attr_value, ast.AST):
            #
            ### Recursively display child AST nodes. ###
            #
            print(f"{attr_prefix}\033[3;33m{field}\033[m \033[31m({attr_type})\033[m:")
            #
            display_ast_tree(attr_value, attr_indent + 4, last_is_list_item=False, parent_is_list=False, continuation_columns=attr_continuation_columns)

        #
        elif isinstance(attr_value, list) and len(cast(list[Any], attr_value)) == 0:
            #
            ### Display lists of AST nodes or other values. ###
            #
            print(f"{attr_prefix}\033[3;33m{field}\033[m \033[31m({attr_type})\033[m: []")

        #
        elif isinstance(attr_value, list):
            #
            ### Display lists of AST nodes or other values. ###
            #
            print(f"{attr_prefix}\033[3;33m{field}\033[m \033[31m({attr_type})\033[m: [")

            #
            ### Update continuation for list items. ###
            #
            list_continuation_columns = attr_continuation_columns.copy()

            #
            j: int
            item: Any
            #
            for j, item in enumerate( cast(list[Any],attr_value) ):

                #
                is_last_item = ( j == (len( cast(list[Any],attr_value) ) - 1) )

                #
                if isinstance(item, ast.AST):
                    #
                    ### Recursively display items in the list.  ###
                    #
                    display_ast_tree(item, attr_indent + 8, last_is_list_item=is_last_item, parent_is_list=True, continuation_columns=list_continuation_columns)
                #
                else:
                    #
                    ### Display non-AST values in the list. ###
                    #
                    item_prefix: str = ""
                    #
                    k: int
                    #
                    for k in range(0, attr_indent + 2, 2):
                        #
                        if k in list_continuation_columns:
                            #
                            item_prefix += "│ "
                        #
                        else:
                            #
                            item_prefix += "  "

                    #
                    if is_last_item:
                        #
                        item_prefix += "└─ "
                    #
                    else:
                        #
                        item_prefix += "├─ "

                    #
                    print(f"{item_prefix}{item!r}")

            #
            ### Closing bracket for list. ###
            #
            closing_prefix: str = ""
            #
            for k in range(0, attr_indent + 4, 2):
                #
                if k in attr_continuation_columns:
                    #
                    closing_prefix += "│ "
                #
                else:
                    #
                    closing_prefix += "  "
            #
            print(f"{closing_prefix}]")

        #
        else:

            #
            ### Display other attributes. ###
            #
            print(f"{attr_prefix}\033[3;33m{field}\033[m \033[31m({attr_type})\033[m: \033[34m{attr_value!r}\033[m")


#
if __name__ == "__main__":

    #
    ### Argument parsing to get custom code. ###
    #
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Parse and display the AST of provided Python code.")
    #
    parser.add_argument("--code", type=str, help="The Python code to parse into an AST.")
    #
    args: argparse.Namespace = parser.parse_args()

    #
    ### Get the custom code from arguments. ###
    #
    with open(args.code, 'r') as file:
        #
        custom_code: str = file.read()

    #
    ### Parse the Python code into an AST. ###
    #
    tree: ast.AST = ast.parse(custom_code)

    #
    print("\n\nParsed AST Tree:\n\n")

    #
    ### Display the parsed AST tree. ###
    #
    display_ast_tree(tree)

    #
    print("\n\n")
