#
import sys
#
sys.path.append( "../1_Conversion_Intermediate_Language_Model_Architecture/" )
#
import lib_classes as lc  # type: ignore


#
linear: lc.ModelBlock = lc.ModelBlock(block_name="Linear")
linear.block_parameters = {
    "input_dim": ("int", None),
    "output_dim": ("int", None),
    "biais": ("int", 1),
    "A": ("Tensor[input_dim, output_dim]", None),
    "b": ("Tensor[1, output_dim]", None)
}
#
linear_forward: lc.BlockFunction = lc.BlockFunction(
    function_name="forward",
    function_arguments={
        "X": ("Tensor[*dims, input_dim]", None),
    },
    model_block=linear
)
#
linear_forward.function_flow_control = [
    #
    lc.FlowControlBasicBinaryOperation(
        output_var_name="X1", input1_var_name="X", operation="*", input2_var_name="A" ),
    #
    lc.FlowControlBasicBinaryOperation(
        output_var_name="B", input1_var_name="biais", operation="*", input2_var_name="b" ),
    #
    lc.FlowControlBasicBinaryOperation(
        output_var_name="Y", input1_var_name="X1", operation="+", input2_var_name="B" ),
    #
    lc.FlowControlReturn( return_variables=["Y"] )
]

#
linear.block_functions = {
    "forward": linear_forward
}




