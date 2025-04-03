import sys
#
sys.path.append( "../../1_Extraction/" )
import lib_classes as lc  # type: ignore

ReLU: lc.ModelBlock = lc.ModelBlock(block_name="ReLU")
ReLU.block_parameters = {
    "inplace": ("bool", False)
}
ReLU_forward: lc.BlockFunction = lc.BlockFunction(
    function_name = "forward", 
    function_arguments = {
        "X": ("Tensor[*dims, input_dim]", None)
    },
    model_block = ReLU
)

ReLU_forward.function_flow_control = [
    lc.FlowControlFunctionCall(
        output_variables=["Y"], function_called="max_tensor_scal", function_arguments={
            "elt1": lc.ExpressionVariable("X"),
            "elt2": lc.ExpressionConstant(0)
        }
    ),
    lc.FlowControlReturn( return_variables=["Y"] )
]

ReLU.block_functions = {
    "forward": ReLU_forward
}