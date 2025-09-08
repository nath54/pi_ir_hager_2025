#
### Import Modules. ###
#
import lib_impl.lib_classes as lc


#
linear: lc.ModelBlock = lc.ModelBlock(block_name="Linear")
linear.block_parameters = {
    "in_features": (lc.VarType("int"), lc.ExpressionNoDefaultArguments()),
    "out_features": (lc.VarType("int"), lc.ExpressionNoDefaultArguments()),
    "bias": (lc.VarType("bool"), lc.ExpressionConstant(True))
}
#
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
        output_var_name="X1", input1_var_name="X", operation="*", input2_var_name="A"
    ),
    #
    lc.FlowControlCondition(
        conditions_fn_call={

            # if bias
            lc.ConditionUnary(elt=lc.ExpressionVariable(var_name="bias")): lc.FlowControlSubBlockFunctionCall(
                output_variables=["Y"], function_called="with_bias", function_arguments={
                    "X": lc.ExpressionVariable(var_name="X1")
                }
            ),

            # else
            lc.ConditionElse(): lc.FlowControlSubBlockFunctionCall(
                output_variables=["Y"], function_called="without_bias", function_arguments={
                    "X": lc.ExpressionVariable(var_name="X1")
                }
            )

        }
    ),
    #
    lc.FlowControlReturn( return_variables=["Y"] )
]

#
linear_with_bias: lc.BlockFunction = lc.BlockFunction(
    function_name="with_bias",
    function_arguments={
        "X": ( lc.VarTypeTensor(tensor_type="number", tensor_dims=["*dims", "input_dim"]), lc.ExpressionNoDefaultArguments()),
    },
    model_block=linear
)
linear_with_bias.function_flow_control = [
    #
    lc.FlowControlBasicBinaryOperation(
        output_var_name="Y", input1_var_name="X", operation="+", input2_var_name="b"
    ),
    #
    lc.FlowControlReturn(
        return_variables=["Y"]
    )
]

#
linear_without_bias: lc.BlockFunction = lc.BlockFunction(
    function_name="without_bias",
    function_arguments={
        "X": ( lc.VarTypeTensor(tensor_type="number", tensor_dims=["*dims", "input_dim"]), lc.ExpressionNoDefaultArguments()),
    },
    model_block=linear
)
linear_without_bias.function_flow_control = [
    #
    lc.FlowControlReturn(
        return_variables=["X"]
    )
]

#
linear.block_functions = {
    "forward": linear_forward,
    "with_bias": linear_with_bias,
    "without_bias": linear_without_bias
}
