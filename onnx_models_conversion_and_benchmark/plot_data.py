#
### Import Modules. ###
#
import json
#
import matplotlib.pyplot as plt
import numpy as np


#
### Load the JSON data. ###
#
with open('saved_models_measurements.json', 'r') as f:
    #
    data = json.load(f)

#
### Extract the relevant dictionaries. ###
#
model_nb_parameters = data['model_nb_parameters']
model_onnx_inference_times = data['model_onnx_inference_times_ms']
model_onnx_ram = data['model_onnx_ram_breakdown_kb']

#
def extract_model_family(model_name: str) -> str:
    """
    Extract the model family from the model name by removing 'model_' prefix
    and identifying the core family name.
    """

    #
    if model_name.startswith('model_'):
        #
        base_name = model_name[6:]
    #
    else:
        #
        base_name = model_name

    #
    ### Known model families from the dataset. ###
    #
    known_families: set[str] = set([
        'simple_lin', 'single_linear', 'mlp_flatten_first', 'parallel_features',
        'factorized', 'global_statistics', 'conv1d_temporal', 'conv1d_feature',
        'conv2d_standard', 'depthwise_separable', 'multiscale_cnn', 'stacked_conv2d',
        'residual_cnn', 'simple_rnn', 'lstm', 'gru', 'bidirectional_lstm',
        'bidirectional_gru', 'self_attention', 'lightweight_transformer',
        'vision_transformer', 'global_avg_pool', 'global_max_pool', 'mixed_pooling',
        'cnn_rnn_hybrid', 'conv_lstm_hybrid', 'cnn_attention', 'tcn', 'senet'
    ])

    #
    ### Exact match. ###
    #
    if base_name in known_families:
        #
        return base_name


    #
    ### Prefix match (longest first to handle multi-word families correctly). ###
    #
    sorted_families = sorted(list(known_families), key=len, reverse=True)
    #
    for family in sorted_families:
        #
        if base_name.startswith(family + '_') or base_name == family:
            #
            return family

    #
    ### Fallback. ###
    #
    return base_name

#
### Group models by family. ###
#
models_by_family: dict[str, list[str]] = {}
#
for model_name in model_nb_parameters.keys():
    #
    family: str = extract_model_family(model_name)
    #
    if not family in models_by_family:
        #
        models_by_family[family] = []
    #
    models_by_family[family].append(model_name)


#
### Prepare data for plotting. ###
#
families = list(models_by_family.keys())

#
### Create a colormap with enough colors for all families. ###
#
colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(families))))
#
if len(families) > 20:
    #
    colors2 = plt.cm.tab20b(np.linspace(0, 1, len(families) - 20))
    #
    colors = np.vstack((colors, colors2))


#
### Figure 1: Nb parameters vs ONNX inference time. ###
#
plt.figure(figsize=(15, 10))
#
for i, family in enumerate(families):
    #
    x_vals = []
    y_vals = []
    #
    for model in models_by_family[family]:
        #
        if model in model_nb_parameters and model in model_onnx_inference_times:
            #
            nb_params = model_nb_parameters[model]
            #
            mean_inference_time = np.mean(model_onnx_inference_times[model])
            #
            x_vals.append(nb_params)
            y_vals.append(mean_inference_time)

    #
    if x_vals:
        #
        color = colors[i] if i < len(colors) else 'black'
        #
        plt.scatter(x_vals, y_vals, color=color, label=family, s=60, alpha=0.7)
        #
        # plt.plot(x_vals, y_vals, color=color, label=family)


#
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel('ONNX Inference Time (ms)', fontsize=12)
plt.title('Figure 1: Number of Parameters vs ONNX Inference Time', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('figure1_nb_params_vs_onnx_inference_time.png', dpi=300, bbox_inches='tight')
plt.show()


#
### Figure 2: Nb parameters vs ONNX RAM. ###
#
plt.figure(figsize=(15, 10))
#
for i, family in enumerate(families):
    #
    x_vals = []
    y_vals = []
    #
    for model in models_by_family[family]:
        #
        if model in model_nb_parameters and model in model_onnx_ram:
            #
            nb_params = model_nb_parameters[model]
            #
            ram_usage = model_onnx_ram[model]['inference_total_kb']
            #
            x_vals.append(nb_params)
            y_vals.append(ram_usage)

    #
    if x_vals:
        #
        color = colors[i] if i < len(colors) else 'black'
        #
        plt.scatter(x_vals, y_vals, color=color, label=family, s=60, alpha=0.7)
        #
        # plt.plot(x_vals, y_vals, color=color, label=family)

#
plt.xlabel('Number of Parameters', fontsize=12)
plt.ylabel('ONNX RAM Usage (KB)', fontsize=12)
plt.title('Figure 2: Number of Parameters vs ONNX RAM Usage', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('figure2_nb_params_vs_onnx_ram.png', dpi=300, bbox_inches='tight')
plt.show()

#
### Print summary. ###
#
print(f"Total models: {len(model_nb_parameters)}")
print(f"Total families: {len(families)}")
print("\nModels per family:")
#
for family in sorted(families):
    #
    count = len(models_by_family[family])
    print(f"  {family}: {count} models")
