# source ../venv/bin/activate
# python lib_weights_link.py "tests/test_model_architecture_1.py" "NO_WEIGHTS"
# python lib_weights_link.py "tests/test_model_architecture_3_1.py" "NO_WEIGHTS"
# python lib_weights_link.py tests/test_model_architecture_final_1.py ../sdia2025/model.pth --main-block=DeepArcNet
python lib_weights_link.py tests/test_model_architecture_final_3.py ../sdia2025/model.pth --main-block=DeepArcNet