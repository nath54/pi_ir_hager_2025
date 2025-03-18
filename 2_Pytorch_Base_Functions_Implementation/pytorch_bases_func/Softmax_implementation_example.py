import sys
#
sys.path.append( "../../1_Extraction/" )
import lib_classes as lc  # type: ignore

Softmax: lc.ModelBlock = lc.ModelBlock(block_name = "Softmax")
Softmax.block_parameters = {
    "dim": ("int", None)
}
Softmax_forward = lc.BlockFunction = lc.BlockFunction(
    function_name = "forward", 
    function_arguments = {
        "X": ("Tensor[*dims, input_dim]", None)
    },
    model_block = Softmax
)

Softmax_forward.function_flow_control = [
    lc.FlowControlSubBlockFunctionCall(output_variables=["Y"], function_called="Softmax", function_arguments={"X": lc.ExpressionVariable("X"), "dim": lc.ExpressionVariable("dim")}),
    lc.FlowControlReturn( return_variables=["Y"])]