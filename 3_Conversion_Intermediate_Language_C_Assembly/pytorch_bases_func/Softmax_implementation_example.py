import sys
#
sys.path.append( "../../1_Conversion_Intermediate_Language_Model_Architecture/" )
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
    
]