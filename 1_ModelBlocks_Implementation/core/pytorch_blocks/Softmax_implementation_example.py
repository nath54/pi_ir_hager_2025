#
### Import Modules. ###
#
import lib_impl.lib_classes as lc


#
Softmax: lc.ModelBlock = lc.ModelBlock(block_name = "Softmax")
Softmax.block_parameters = {
    "dim": (lc.VarType("int"), lc.ExpressionNoDefaultArguments())
}
Softmax_forward = lc.BlockFunction = lc.BlockFunction(
    function_name = "forward",
    function_arguments = {
        "X": ( lc.VarTypeTensor(tensor_type="number", tensor_dims=["*dims", "input_dim"]), lc.ExpressionNoDefaultArguments()),
    },
    model_block = Softmax
)

Softmax_forward.function_flow_control = [
    lc.FlowControlSubBlockFunctionCall(output_variables=["Y"], function_called="Softmax", function_arguments={"X": lc.ExpressionVariable("X"), "dim": lc.ExpressionVariable("dim")}),
    lc.FlowControlReturn( return_variables=["Y"])]