#
### Import Modules. ###
#
from typing import Any
#
import torch
from torch import Tensor
from torch import nn
#
from lib_import_pt_models import import_module_from_filepath
#
from lib_test import Model_Processing_and_Tester


#
###
#
class DataModel:

    #
    def __init__(
        self,
        model_name: str,
        model_family: int,
        pytorch_file_path: str,
        model_kwargs: dict[str, Any],
        main_pt_class_name: str = "Model"
    ) -> None:

        #
        self.model_name: str = model_name
        self.model_family: int = model_family
        self.pytorch_file_path: str = pytorch_file_path
        self.main_pt_class_name: str = main_pt_class_name
        #
        self.model_kwargs: dict[str, Any] = model_kwargs


#
### Data models Class. ###
#
class DataModelsClass:

    #
    def __init__(self) -> None:

        #
        self.models_to_test: list[DataModel] = [

            #
            ### ==== Linear Models. ==== ###
            #

            #
            ## Simple Linears. ##
            #
            DataModel(model_name="model_simple_lin_1",              model_family=0,     pytorch_file_path="model_00_simple_lin_1.py",                  model_kwargs={'h0': 16}),  # Num Params=657; ONNX_inf=0.182ms ± 0.057ms; 4.48KB
            DataModel(model_name="model_simple_lin_2",              model_family=0,     pytorch_file_path="model_01_simple_lin_2.py",                  model_kwargs={'h0': 16, 'h1': 32}),  # Num Params=15601; ONNX_inf=0.163ms ± 0.015ms; ONNX total RAM=63.19KB
            DataModel(model_name="model_simple_lin_N_3",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 16]}),  # Num Params=16113; ONNX_inf=0.172ms ± 0.017ms; ONNX total RAM=65.52KB
            DataModel(model_name="model_simple_lin_N_4",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 16]}),  # Num Params=18737; ONNX_inf=0.203ms ± 0.053ms; ONNX total RAM=76.10KB
            DataModel(model_name="model_simple_lin_N_5",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 32, 16]}),  # Num Params=20305; ONNX_inf=0.185ms ± 0.056ms; ONNX total RAM=82.57KB
            DataModel(model_name="model_simple_lin_N_6",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 32, 16]}),  # Num Params=30673; ONNX_inf=0.196ms ± 0.034ms; ONNX total RAM=123.42KB
            DataModel(model_name="model_simple_lin_N_7",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 64, 32, 16]}),  # Num Params=36881; ONNX_inf=0.215ms ± 0.028ms; ONNX total RAM=148.01KB
            DataModel(model_name="model_simple_lin_N_8",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [32, 64, 128, 256, 128, 64, 32]}),
            DataModel(model_name="model_simple_lin_N_9",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [64, 128, 256, 512, 256, 128, 64]}),

            #
            ## Most Basic Perceptron Model. ##
            #
            DataModel(model_name="model_single_linear",             model_family=1,     pytorch_file_path="model_03_single_layer_perceptron.py",       model_kwargs={}),  # Num Params=301; ONNX_inf=0.134ms ± 0.014ms; ONNX total RAM=2.76KB

            #
            ## Mlp flatten first. ##
            #
            DataModel(model_name="model_mlp_flatten_first",         model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 32, 'depth': 2}),  # Num Params=10721; ONNX_inf=0.161ms ± 0.045ms; ONNX total RAM=44.13KB
            DataModel(model_name="model_mlp_flatten_first_d3",      model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 32, 'depth': 3}),
            DataModel(model_name="model_mlp_flatten_first_d4",      model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 32, 'depth': 4}),
            DataModel(model_name="model_mlp_flatten_first_h64_d2",  model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_mlp_flatten_first_h64_d4",  model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 64, 'depth': 4}),
            DataModel(model_name="model_mlp_flatten_first_h128_d3", model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 128, 'depth': 3}),

            #
            ## MLP parallel features. ##
            #
            DataModel(model_name="model_parallel_features",         model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 16, 'depth': 2}),  # Num Params=945; ONNX_inf=0.189ms ± 0.009ms; ONNX total RAM=6.55KB
            DataModel(model_name="model_parallel_features_d3",      model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 16, 'depth': 3}),
            DataModel(model_name="model_parallel_features_h32_d2",  model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_parallel_features_h32_d3",  model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 32, 'depth': 3}),
            DataModel(model_name="model_parallel_features_h64_d2",  model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_parallel_features_h64_d4",  model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 64, 'depth': 4}),

            #
            ## MLP factorized. ##
            #
            DataModel(model_name="model_factorized",                model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 8, 'depth': 1}),  # Num Params=128; ONNX_inf=0.205ms ± 0.076ms; ONNX total RAM=2.91KB
            DataModel(model_name="model_factorized_d2",             model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 8, 'depth': 2}),
            DataModel(model_name="model_factorized_h16_d1",         model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 16, 'depth': 1}),
            DataModel(model_name="model_factorized_h16_d2",         model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_factorized_h32_d2",         model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_factorized_h64_d3",         model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 64, 'depth': 3}),

            #
            ## MLP Based on global statistics extraction. ##
            #
            DataModel(model_name="model_global_statistics",         model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 16, 'depth': 1}),  # Num Params=673; ONNX_inf=0.206ms ± 0.022ms; ONNX total RAM=5.80KB
            DataModel(model_name="model_global_statistics_d2",      model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_global_statistics_h32_d1",  model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 32, 'depth': 1}),
            DataModel(model_name="model_global_statistics_h32_d2",  model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_global_statistics_h64_d2",  model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_global_statistics_h128_d3", model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 128, 'depth': 3}),

            #
            ### ==== Conv Models. ==== ###
            #

            #
            ## Conv1d temporal. ##
            #
            DataModel(model_name="model_conv1d_temporal",           model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 8, 'kernel_size': 3, 'pool_size': 2, 'depth': 3}),  # Num Params=665; ONNX_inf=0.270ms ± 0.102ms; ONNX total RAM=5.80KB
            DataModel(model_name="model_conv1d_temporal_d4",        model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 8, 'kernel_size': 3, 'pool_size': 1, 'depth': 4}),
            DataModel(model_name="model_conv1d_temporal_c16_d3",    model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 16, 'kernel_size': 3, 'pool_size': 1, 'depth': 3}),
            DataModel(model_name="model_conv1d_temporal_c16_d5",    model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 16, 'kernel_size': 3, 'pool_size': 1, 'depth': 5}),
            DataModel(model_name="model_conv1d_temporal_c32_d4",    model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 32, 'kernel_size': 3, 'pool_size': 1, 'depth': 4}),
            DataModel(model_name="model_conv1d_temporal_c32_d6",    model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 32, 'kernel_size': 5, 'pool_size': 1, 'depth': 6}),

            #
            ## Conv1d feature. ##
            #
            DataModel(model_name="model_conv1d_feature",            model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 8, 'k0': 3, 'depth': 1}),  # Num Params=1953; ONNX_inf=0.216ms ± 0.055ms; ONNX total RAM=10.03KB
            DataModel(model_name="model_conv1d_feature_d2",         model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 8, 'k0': 3, 'depth': 2}),
            DataModel(model_name="model_conv1d_feature_c16_d1",     model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 16, 'k0': 3, 'depth': 1}),
            DataModel(model_name="model_conv1d_feature_c16_d2",     model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 16, 'k0': 3, 'depth': 2}),
            DataModel(model_name="model_conv1d_feature_c32_d3",     model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 32, 'k0': 5, 'depth': 3}),
            DataModel(model_name="model_conv1d_feature_c64_d4",     model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 64, 'k0': 5, 'depth': 4}),

            #
            ## Conv2d standard. ##
            #
            DataModel(model_name="model_conv2d_standard",           model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 3}),  # Num Params=4647; ONNX_inf=0.389ms ± 0.240ms; ONNX total RAM=21.50KB
            DataModel(model_name="model_conv2d_standard_d4",        model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 4}),
            DataModel(model_name="model_conv2d_standard_c16_d3",    model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 3}),
            DataModel(model_name="model_conv2d_standard_c16_d5",    model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 5}),
            DataModel(model_name="model_conv2d_standard_c32_d4",    model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 32, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 4}),
            DataModel(model_name="model_conv2d_standard_c32_d6",    model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 32, 'k_h': 5, 'k_w': 5, 'p': 2, 'depth': 6}),

            #
            ## Conv2d depthwise separable. ##
            #
            DataModel(model_name="model_depthwise_separable",       model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 2, 'depth': 1}),  # Num Params=475; ONNX_inf=0.237ms ± 0.099ms; ONNX total RAM=4.45KB
            DataModel(model_name="model_depthwise_separable_d2",    model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 2, 'depth': 2}),
            DataModel(model_name="model_depthwise_separable_c16_d1",model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'p': 2, 'depth': 1}),
            DataModel(model_name="model_depthwise_separable_c16_d2",model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'p': 2, 'depth': 2}),
            DataModel(model_name="model_depthwise_separable_c32_d3",model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 32, 'k_h': 5, 'k_w': 5, 'p': 1, 'depth': 3}),
            DataModel(model_name="model_depthwise_separable_c64_d4",model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 64, 'k_h': 5, 'k_w': 5, 'p': 1, 'depth': 4}),

            #
            ## Multiscale CNN. ##
            #
            DataModel(model_name="model_multiscale_cnn",            model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 4, 'depth': 1}),  # Num Params=357; ONNX_inf=0.289ms ± 0.045ms; ONNX total RAM=4.83KB
            DataModel(model_name="model_multiscale_cnn_d2",         model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 4, 'depth': 2}),
            DataModel(model_name="model_multiscale_cnn_c8_d1",      model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 8, 'depth': 1}),
            DataModel(model_name="model_multiscale_cnn_c8_d2",      model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 8, 'depth': 2}),
            DataModel(model_name="model_multiscale_cnn_c16_d3",     model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 16, 'depth': 3}),
            DataModel(model_name="model_multiscale_cnn_c32_d4",     model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 32, 'depth': 4}),

            #
            ## Stacked Conv2d. ##
            #
            DataModel(model_name="model_stacked_conv2d",            model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 8, 'c1': 16, 'k0': 3, 'k1': 3, 'p': 2, 'depth': 2}),  # Num Params=1265; ONNX_inf=0.232ms ± 0.059ms; ONNX total RAM=7.84KB
            DataModel(model_name="model_stacked_conv2d_d3",         model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 8, 'c1': 16, 'k0': 3, 'k1': 3, 'p': 2, 'depth': 3}),
            DataModel(model_name="model_stacked_conv2d_c16_c32_d2", model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 16, 'c1': 32, 'k0': 3, 'k1': 3, 'p': 2, 'depth': 2}),
            DataModel(model_name="model_stacked_conv2d_c16_c32_d3", model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 16, 'c1': 32, 'k0': 3, 'k1': 3, 'p': 2, 'depth': 3}),
            DataModel(model_name="model_stacked_conv2d_c32_c64_d3", model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 32, 'c1': 64, 'k0': 5, 'k1': 5, 'p': 1, 'depth': 3}),
            DataModel(model_name="model_stacked_conv2d_c64_c128_d4",model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 64, 'c1': 128, 'k0': 5, 'k1': 5, 'p': 1, 'depth': 4}),

            #
            ## Residual CNN. ##
            #
            DataModel(model_name="model_residual_cnn",              model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'depth': 2}),  # Num Params=2425; ONNX_inf=0.631ms ± 0.900ms; ONNX total RAM=13.33KB
            DataModel(model_name="model_residual_cnn_d3",           model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'depth': 3}),
            DataModel(model_name="model_residual_cnn_c16_d2",       model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'depth': 2}),
            DataModel(model_name="model_residual_cnn_c16_d4",       model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'depth': 4}),
            DataModel(model_name="model_residual_cnn_c32_d3",       model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 32, 'k_h': 3, 'k_w': 3, 'depth': 3}),
            DataModel(model_name="model_residual_cnn_c64_d5",       model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 64, 'k_h': 5, 'k_w': 5, 'depth': 5}),

            #
            ### ==== Recurrent Models. ==== ###
            #

            #
            ## Simple RNN. ##
            #
            DataModel(model_name="model_simple_rnn",                model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 16, 'depth': 1}),  # Num Params=465; ONNX_inf=0.199ms ± 0.009ms; ONNX total RAM=5.02KB
            DataModel(model_name="model_simple_rnn_d2",             model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_simple_rnn_h32_d1",         model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 32, 'depth': 1}),
            DataModel(model_name="model_simple_rnn_h32_d2",         model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_simple_rnn_h64_d2",         model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_simple_rnn_h128_d3",        model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 128, 'depth': 3}),

            #
            ## Simple LSTM. ##
            #
            DataModel(model_name="model_lstm",                      model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 16, 'depth': 1}),  # Num Params=1809; ONNX_inf=0.211ms ± 0.042ms; ONNX total RAM=10.99KB
            DataModel(model_name="model_lstm_d2",                   model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_lstm_h32_d1",               model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 32, 'depth': 1}),
            DataModel(model_name="model_lstm_h32_d2",               model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_lstm_h64_d2",               model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_lstm_h128_d3",              model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 128, 'depth': 3}),

            #
            ## Simple GRU. ##
            #
            DataModel(model_name="model_gru",                       model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 16, 'depth': 1}),  # Num Params=1361; ONNX_inf=0.238ms ± 0.108ms; ONNX total RAM=8.53KB
            DataModel(model_name="model_gru_d2",                    model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_gru_h32_d1",                model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 32, 'depth': 1}),
            DataModel(model_name="model_gru_h32_d2",                model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_gru_h64_d2",                model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 64, 'depth': 2}),
            DataModel(model_name="model_gru_h128_d3",               model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 128, 'depth': 3}),

            #
            ## Bidirectional LSTM. ##
            #
            DataModel(model_name="model_bidirectional_lstm",        model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 8, 'depth': 1}),  # Num Params=1297; ONNX_inf=0.253ms ± 0.036ms; ONNX total RAM=9.15KB
            DataModel(model_name="model_bidirectional_lstm_d2",     model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 8, 'depth': 2}),
            DataModel(model_name="model_bidirectional_lstm_h16_d1", model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 16, 'depth': 1}),
            DataModel(model_name="model_bidirectional_lstm_h16_d2", model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_bidirectional_lstm_h32_d2", model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_bidirectional_lstm_h64_d3", model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 64, 'depth': 3}),

            #
            ## Bidirectional GRU. ##
            #
            DataModel(model_name="model_bidirectional_gru",         model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 8, 'depth': 1}),  # Num Params=977; ONNX_inf=0.252ms ± 0.035ms; ONNX total RAM=7.19KB
            DataModel(model_name="model_bidirectional_gru_d2",      model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 8, 'depth': 2}),
            DataModel(model_name="model_bidirectional_gru_h16_d1",  model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 16, 'depth': 1}),
            DataModel(model_name="model_bidirectional_gru_h16_d2",  model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 16, 'depth': 2}),
            DataModel(model_name="model_bidirectional_gru_h32_d2",  model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_bidirectional_gru_h64_d3",  model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 64, 'depth': 3}),

            #
            ### ==== Attention based Models. ==== ###
            #

            #
            ## Self Attention Model. ##
            #
            DataModel(model_name="model_self_attention",            model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 16, 'depth': 1}),  # Num Params=1313; ONNX_inf=0.249ms ± 0.034ms; ONNX total RAM=9.25KB
            DataModel(model_name="model_self_attention_d2",         model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 16, 'depth': 2}),
            DataModel(model_name="model_self_attention_dk32_d1",    model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 32, 'depth': 1}),
            DataModel(model_name="model_self_attention_dk32_d2",    model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 32, 'depth': 2}),
            DataModel(model_name="model_self_attention_dk64_d2",    model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 64, 'depth': 2}),
            DataModel(model_name="model_self_attention_dk128_d3",   model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 128, 'depth': 3}),

            #
            ## Transformer Model. ##
            #
            DataModel(model_name="model_lightweight_transformer",               model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 16, 'num_heads': 1, 'depth': 1}),  # Num Params=3473; ONNX_inf=0.418ms ± 0.123ms; ONNX total RAM=21.42KB
            DataModel(model_name="model_lightweight_transformer_d2",            model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 16, 'num_heads': 1, 'depth': 2}),
            DataModel(model_name="model_lightweight_transformer_dm32_nh2_d1",   model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 32, 'num_heads': 2, 'depth': 1}),
            DataModel(model_name="model_lightweight_transformer_dm32_nh2_d2",   model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 32, 'num_heads': 2, 'depth': 2}),
            DataModel(model_name="model_lightweight_transformer_dm64_nh4_d2",   model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 64, 'num_heads': 4, 'depth': 2}),
            DataModel(model_name="model_lightweight_transformer_dm128_nh8_d3",  model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",        model_kwargs={'d_model': 128, 'num_heads': 8, 'depth': 3}),

            #
            ## Vision Transformer Model. ##
            #
            DataModel(model_name="model_vision_transformer",                model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 16, 'num_heads': 2, 'depth': 2, 'mlp_ratio': 4}),  # Num Params=7489; ONNX_inf=0.465ms ± 0.047ms; ONNX total RAM=44.48KB
            DataModel(model_name="model_vision_transformer_d3",             model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 16, 'num_heads': 2, 'depth': 3, 'mlp_ratio': 4}),
            DataModel(model_name="model_vision_transformer_dm32_d2",        model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 32, 'num_heads': 4, 'depth': 2, 'mlp_ratio': 4}),
            DataModel(model_name="model_vision_transformer_dm32_d4",        model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 32, 'num_heads': 4, 'depth': 4, 'mlp_ratio': 4}),
            DataModel(model_name="model_vision_transformer_dm64_d3",        model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 64, 'num_heads': 8, 'depth': 3, 'mlp_ratio': 4}),
            DataModel(model_name="model_vision_transformer_ps4_dm128_d4",   model_family=20,    pytorch_file_path="model_22_vision_transformer.py",             model_kwargs={'patch_size': 5, 'd_model': 128, 'num_heads': 8, 'depth': 4, 'mlp_ratio': 4}),

            #
            ### ==== Pooling Models. ==== ###
            #

            #
            ## Global AVG Pooling Model. ##
            #
            DataModel(model_name="model_global_avg_pool",           model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 16, 'depth': 2}),  # Num Params=465; ONNX_inf=0.207ms ± 0.074ms; ONNX total RAM=4.32KB
            DataModel(model_name="model_global_avg_pool_d3",        model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 16, 'depth': 3}),
            DataModel(model_name="model_global_avg_pool_h32_d2",    model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_global_avg_pool_h32_d3",    model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 32, 'depth': 3}),
            DataModel(model_name="model_global_avg_pool_h64_d3",    model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 64, 'depth': 3}),
            DataModel(model_name="model_global_avg_pool_h128_d4",   model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 128, 'depth': 4}),

            #
            ## Global Max Pooling Model. ##
            #
            DataModel(model_name="model_global_max_pool",           model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 16, 'depth': 2}),  # Num Params=465; ONNX_inf=0.184ms ± 0.021ms; ONNX total RAM=4.33KB
            DataModel(model_name="model_global_max_pool_d3",        model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 16, 'depth': 3}),
            DataModel(model_name="model_global_max_pool_h32_d2",    model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_global_max_pool_h32_d3",    model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 32, 'depth': 3}),
            DataModel(model_name="model_global_max_pool_h64_d3",    model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 64, 'depth': 3}),
            DataModel(model_name="model_global_max_pool_h128_d4",   model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 128, 'depth': 4}),

            #
            ## Global Mixed Pooling Model. ##
            #
            DataModel(model_name="model_mixed_pooling",             model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 16, 'depth': 2}),  # Num Params=625; ONNX_inf=0.206ms ± 0.022ms; ONNX total RAM=5.29KB
            DataModel(model_name="model_mixed_pooling_d3",          model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 16, 'depth': 3}),
            DataModel(model_name="model_mixed_pooling_h32_d2",      model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 32, 'depth': 2}),
            DataModel(model_name="model_mixed_pooling_h32_d3",      model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 32, 'depth': 3}),
            DataModel(model_name="model_mixed_pooling_h64_d3",      model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 64, 'depth': 3}),
            DataModel(model_name="model_mixed_pooling_h128_d4",     model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 128, 'depth': 4}),

            #
            ### ==== Hybrid / Specialized Models. ==== ###
            #

            #
            ## CNN / RNN Hybrid. ##
            #
            DataModel(model_name="model_cnn_rnn_hybrid",            model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 8, 'k0': 3, 'h0': 3, 'depth': 1}),  # Num Params=369; ONNX_inf=0.308ms ± 0.082ms; ONNX total RAM=5.05KB
            DataModel(model_name="model_cnn_rnn_hybrid_d2",         model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 8, 'k0': 3, 'h0': 3, 'depth': 2}),
            DataModel(model_name="model_cnn_rnn_hybrid_c16_h16_d1", model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 16, 'k0': 3, 'h0': 16, 'depth': 1}),
            DataModel(model_name="model_cnn_rnn_hybrid_c16_h16_d2", model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 16, 'k0': 3, 'h0': 16, 'depth': 2}),
            DataModel(model_name="model_cnn_rnn_hybrid_c32_h32_d2", model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 32, 'k0': 5, 'h0': 32, 'depth': 2}),
            DataModel(model_name="model_cnn_rnn_hybrid_c64_h64_d3", model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 64, 'k0': 5, 'h0': 64, 'depth': 3}),

            #
            ## Conv / LSTM hybrid. ##
            #
            DataModel(model_name="model_conv_lstm_hybrid",              model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 8, 'k0': 3, 'h0': 12, 'depth': 1}),  # Num Params=1317; ONNX_inf=0.372ms ± 0.236ms; ONNX total RAM=9.51KB
            DataModel(model_name="model_conv_lstm_hybrid_d2",           model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 8, 'k0': 3, 'h0': 12, 'depth': 2}),
            DataModel(model_name="model_conv_lstm_hybrid_c16_h32_d1",   model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 16, 'k0': 3, 'h0': 32, 'depth': 1}),
            DataModel(model_name="model_conv_lstm_hybrid_c16_h32_d2",   model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 16, 'k0': 3, 'h0': 32, 'depth': 2}),
            DataModel(model_name="model_conv_lstm_hybrid_c32_h64_d2",   model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 32, 'k0': 5, 'h0': 64, 'depth': 2}),
            DataModel(model_name="model_conv_lstm_hybrid_c64_h128_d3",  model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",             model_kwargs={'c0': 64, 'k0': 5, 'h0': 128, 'depth': 3}),

            #
            ## CNN / Attention hybrid. ##
            #
            DataModel(model_name="model_cnn_attention",             model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'd_k': 12, 'depth': 1}),  # Num Params=417; ONNX_inf=0.404ms ± 0.048ms; ONNX total RAM=5.35KB
            DataModel(model_name="model_cnn_attention_d2",          model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'd_k': 12, 'depth': 2}),
            DataModel(model_name="model_cnn_attention_c16_dk32_d1", model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'd_k': 32, 'depth': 1}),
            DataModel(model_name="model_cnn_attention_c16_dk32_d2", model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'd_k': 32, 'depth': 2}),
            DataModel(model_name="model_cnn_attention_c32_dk64_d2", model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 32, 'k_h': 5, 'k_w': 5, 'd_k': 64, 'depth': 2}),
            DataModel(model_name="model_cnn_attention_c64_dk128_d3",model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 64, 'k_h': 5, 'k_w': 5, 'd_k': 128, 'depth': 3}),

            #
            ## TCN Model. ##
            #
            DataModel(model_name="model_tcn",                       model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 8, 'kernel_size': 3, 'num_layers': 3}),  # Num Params=657; ONNX_inf=0.268ms ± 0.085ms; ONNX total RAM=6.84KB
            DataModel(model_name="model_tcn_l4",                    model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 8, 'kernel_size': 3, 'num_layers': 4}),
            DataModel(model_name="model_tcn_c16_l3",                model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 16, 'kernel_size': 3, 'num_layers': 3}),
            DataModel(model_name="model_tcn_c16_l5",                model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 16, 'kernel_size': 3, 'num_layers': 5}),
            DataModel(model_name="model_tcn_c32_l4",                model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 32, 'kernel_size': 3, 'num_layers': 4}),
            DataModel(model_name="model_tcn_c64_l6",                model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 64, 'kernel_size': 5, 'num_layers': 6}),

            #
            ## Senet Model. ##
            #
            DataModel(model_name="model_senet",                     model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4, 'depth': 1}),  # Num Params=1915; ONNX_inf=0.233ms ± 0.048ms; ONNX total RAM=11.06KB
            DataModel(model_name="model_senet_d2",                  model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4, 'depth': 2}),
            DataModel(model_name="model_senet_c16_d1",              model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4, 'depth': 1}),
            DataModel(model_name="model_senet_c16_d2",              model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 16, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4, 'depth': 2}),
            DataModel(model_name="model_senet_c32_d2",              model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 32, 'k_h': 5, 'k_w': 5, 'reduction_ratio': 8, 'depth': 2}),
            DataModel(model_name="model_senet_c64_d3",              model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 64, 'k_h': 5, 'k_w': 5, 'reduction_ratio': 8, 'depth': 3}),
        ]

    #
    def load_model(self, model_idx: int) -> nn.Module:

        #
        model_data: DataModel = self.models_to_test[model_idx]

        #
        obj: object = import_module_from_filepath(filepath=f"models_py/{model_data.pytorch_file_path}")

        #
        if not hasattr(obj, model_data.main_pt_class_name):
            #
            raise UserWarning(f"Error")

        #
        return getattr(obj, model_data.main_pt_class_name)(**model_data.model_kwargs)


#
### Main Class. ###
#
class MainTestAndMeasures:

    #
    def __init__(self) -> None:

        #
        self.data_models: DataModelsClass = DataModelsClass()

        #
        self.mpt: Model_Processing_and_Tester = Model_Processing_and_Tester()


    #
    def test_one_model(self, model_idx: int) -> None:

        #
        model_name: str = self.data_models.models_to_test[model_idx].model_name

        #
        print(f"\n\n==== Processing model `{model_name}` with parameters: {self.data_models.models_to_test[model_idx].model_kwargs} ====\n\n")

        #
        pt_model: nn.Module = self.data_models.load_model(model_idx=model_idx)

        #
        ### Test model inference before converting into ONNX. ###
        #
        input_tensor: Tensor = torch.randn(self.mpt.input_shape)
        #
        try:
            #
            _ = pt_model(input_tensor)
        #
        except:
            #
            print(f"Error with model : `{model_name}`")
            #
            return

        #
        self.mpt.convert_to_onnx(
            model_name=model_name,
            pt_model=pt_model,
            onnx_filepath=f"models_onnx/{model_name}.onnx"
        )

        #
        self.mpt.measure_model(
            model_name=model_name,
            pt_model=pt_model
        )

        #
        self.mpt.models_families[model_name] = self.data_models.models_to_test[model_idx].model_family


    #
    def main(self) -> None:

        #
        for model_idx in range(len(self.data_models.models_to_test)):

            #
            self.test_one_model(model_idx=model_idx)

            # #
            # try:
            #     #
            #     self.test_one_model(model_idx=model_idx)
            # #
            # except:
            #     #
            #     pass

        #
        self.mpt.save_logs()


#
if __name__ == "__main__":

    #
    main: MainTestAndMeasures = MainTestAndMeasures()

    #
    main.main()
