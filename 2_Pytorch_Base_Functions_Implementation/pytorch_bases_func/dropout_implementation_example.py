import sys
sys.path.append("../../1_Extraction/")
import lib_classes as lc  # type: ignore

############################
# Définition du bloc Dropout
############################
Dropout: lc.ModelBlock = lc.ModelBlock(block_name="Dropout")

# Paramètres : p = probabilité de drop, training = booléen
Dropout.block_parameters = {
    "p": ("float", 0.5),      # probabilité de masquage (entre 0.0 et 1.0)
    "training": ("bool", True)
}

# Fonction forward
Dropout_forward = lc.BlockFunction(
    function_name="forward",
    function_arguments={
        "X": ("Tensor[*dims, input_dim]", None)
    },
    model_block=Dropout
)

# Sub-fonction : applique le masque (Dropout)
apply_mask_fn = lc.BlockFunction(
    function_name="apply_mask",
    function_arguments={
        "X": ("Tensor", None)
    },
    model_block=Dropout
)

# Sub-fonction : pas de drop, retourne l'entrée telle quelle
no_drop_fn = lc.BlockFunction(
    function_name="no_drop",
    function_arguments={
        "X": ("Tensor", None)
    },
    model_block=Dropout
)

###########################
# Corps de apply_mask_fn
#
# 1) Générer un masque binaire ~ Bernoulli(1 - p)
# 2) Appliquer le masque : X * mask
# 3) Redimensionner la sortie par 1 / (1 - p) (optionnel)
#
apply_mask_fn.function_flow_control = [
    # mask = bernoulli(prob = 1 - p)
    lc.FlowControlFunctionCall(
        output_variables=["mask"],
        function_called="bernoulli",
        function_arguments={
            # On calcule (1 - p) directement via une ExpressionToEvaluate
            "prob": lc.ExpressionToEvaluate("1.0 - p"),
            "shape": lc.ExpressionToEvaluate("X.shape")
        }
    ),

    # X_drop = X * mask
    lc.FlowControlBasicBinaryOperation(
        output_var_name="X_drop",
        input1_var_name="X",
        operation="*",
        input2_var_name="mask"
    ),

    # On peut échelle la sortie par (1 / (1 - p)) pour conserver l'espérance
    lc.FlowControlFunctionCall(
        output_variables=["Y"],
        function_called="mul_scalar", # Hypothétique fonction de multiplication par un scalaire
        function_arguments={
            "X": lc.ExpressionVariable(var_name="X_drop"),
            # ExpressionToEvaluate("1.0 / (1.0 - p)")
            "scalar": lc.ExpressionToEvaluate("1.0 / (1.0 - p)")
        }
    ),

    # Retour
    lc.FlowControlReturn(
        return_variables=["Y"]
    )
]

# Corps de no_drop_fn (retourne X directement)
no_drop_fn.function_flow_control = [
    lc.FlowControlReturn(return_variables=["X"])
]

###############################
# Corps de la fonction forward
#
# Si training, on applique le masque, sinon on renvoie X.
#
Dropout_forward.function_flow_control = [
    lc.FlowControlCondition(
        conditions_fn_call={
            lc.ConditionUnary(elt=lc.ExpressionVariable(var_name="training")):
                lc.FlowControlSubBlockFunctionCall(
                    output_variables=["Y"],
                    function_called="apply_mask",
                    function_arguments={
                        "X": lc.ExpressionVariable(var_name="X")
                    }
                ),
            lc.ConditionElse():
                lc.FlowControlSubBlockFunctionCall(
                    output_variables=["Y"],
                    function_called="no_drop",
                    function_arguments={
                        "X": lc.ExpressionVariable(var_name="X")
                    }
                )
        }
    ),

    # Retour final
    lc.FlowControlReturn(
        return_variables=["Y"]
    )
]

################################
# On ajoute les fonctions au bloc
Dropout.block_functions = {
    "forward": Dropout_forward,
    "apply_mask": apply_mask_fn,
    "no_drop": no_drop_fn
}
