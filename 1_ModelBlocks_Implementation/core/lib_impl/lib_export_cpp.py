#
### Import Modules. ###
#
import os
#
import lib_classes as lc
import lib_layers as ll


#
### TODO: All layers will have an efficient C/C++ implementation at path `1_ModelBlocks_Implementation/core/cpp_modules/` ###
#

#
class Language_Model_Exporter:

    #
    def __init__(
        self,
        l_model: lc.Language_Model,
        export_path: str
    ) -> None:

        #
        self.l_model: lc.Language_Model = l_model
        #
        self.export_path: str = export_path


    #
    def export_expression(self, expr: lc.Expression) -> str:

        #
        return "// TODO"

    #
    def export_instruction(self, instruction: lc.FlowControlInstruction) -> str:

        #
        return "// TODO"


    #
    def export_module_block_function(self, mb_fn: lc.BlockFunction) -> str:

        #
        return "// TODO"


    #
    def export_module_block(self, module_block: lc.ModelBlock) -> str:

        #
        return f"""
// TODO
"""

    #
    def export_l_model(self) -> str:

        #
        return f"""
// TODO
        """


    #
    def export_to_cpp(self) -> None:

        #
        if not os.path.exists(self.export_path):
            #
            os.makedirs(self.export_path)

        #
        ### TODO: create code project, MakeFile, build.sh, run.sh, Readme.md. ###
        #
        pass

