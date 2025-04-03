import sys
#
sys.path.append("../../1_Extraction/")
import lib_classes as lc  # type: ignore

# Définition du bloc Softmax
Softmax: lc.ModelBlock = lc.ModelBlock(block_name="Softmax")
Softmax.block_parameters = {
    "dim": ("int", None)  # Paramètre pour définir l'axe de normalisation
}

# Définition de la fonction forward
Softmax_forward = lc.BlockFunction(
    function_name="forward", 
    function_arguments={
        "X": ("Tensor[*dims, input_dim]", None)
    },
    model_block=Softmax
)

# Définition des instructions de contrôle de flux pour forward
Softmax_forward.function_flow_control = [
    # Exponentiation des éléments de X
    lc.FlowControlFunctionCall(
        output_variables=["X_exp"],
        function_called="exp",
        function_arguments={"X": lc.ExpressionVariable(var_name="X")}
    ),
    
    # Calcul de la somme des exponentielles sur l'axe défini
    lc.FlowControlFunctionCall(
        output_variables=["sum_exp"],
        function_called="sum",
        function_arguments={
            "X": lc.ExpressionVariable(var_name="X_exp"),
            "dim": lc.ExpressionVariable(var_name="dim"),
            "keepdim": lc.ExpressionConstantNumeric(True)
        }
    ),
    
    # Division élément par élément
    lc.FlowControlBasicBinaryOperation(
        output_var_name="Y",
        input1_var_name="X_exp",
        operation="/",
        input2_var_name="sum_exp"
    ),
    
    # Retour de Y
    lc.FlowControlReturn(
        return_variables=["Y"]
    )
]

# Ajout de la fonction forward au bloc Softmax
Softmax.block_functions = {
    "forward": Softmax_forward
}
