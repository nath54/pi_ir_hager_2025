# source ../venv/bin/activate
# python script_test_link_weights_to_architecture.py "tests/test_model_architecture_1.py" "NO_WEIGHTS"
# python script_test_link_weights_to_architecture.py "tests/test_model_architecture_3_1.py" "NO_WEIGHTS"
# python script_test_link_weights_to_architecture.py tests/test_model_architecture_final_1.py ../sdia2025/model.pth --main-block=DeepArcNet
python script_test_link_weights_to_architecture.py tests/test_model_architecture_final_3.py ../sdia2025/model.pth --main-block=DeepArcNet