#
### Import Modules. ###
#
import lib_impl.lib_classes as lc


#
linear: lc.ModelBlock = lc.ModelBlock(block_name="Linear")
linear.block_parameters = {
    "in_features": (lc.VarType("int"), lc.ExpressionNoDefaultArguments()),
    "out_features": (lc.VarType("int"), lc.ExpressionNoDefaultArguments()),
    "bias": (lc.VarType("int"), lc.ExpressionConstant(1))
}
linear.block_weights = {
    "A": ["in_features", "out_features"],
    "b": [1, "out_features"]
}
#
linear_forward: lc.BlockFunction = lc.BlockFunction(
    function_name="forward",
    function_arguments={
        "X": ( lc.VarTypeTensor(tensor_type="number", tensor_dims=["*dims", "input_dim"]), lc.ExpressionNoDefaultArguments()),
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
        output_var_name="B", input1_var_name="bias", operation="*", input2_var_name="b" ),
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

