#
### Import Modules. ###
#
import lib_impl.lib_classes as lc


#
ReLU: lc.ModelBlock = lc.ModelBlock(block_name="ReLU")
ReLU.block_parameters = {
    "inplace": ( lc.VarType("bool"), lc.ExpressionConstant(False) )
}
ReLU_forward: lc.BlockFunction = lc.BlockFunction(
    function_name = "forward",
    function_arguments = {
        "X": ( lc.VarTypeTensor(tensor_type="number", tensor_dims=["*dims", "input_dim"]), lc.ExpressionNoDefaultArguments()),
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