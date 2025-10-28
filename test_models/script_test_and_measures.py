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
            DataModel(model_name="model_simple_lin_1",              model_family=0,     pytorch_file_path="model_00_simple_lin_1.py",                  model_kwargs={'h0': 16}),
            DataModel(model_name="model_simple_lin_2",              model_family=0,     pytorch_file_path="model_01_simple_lin_2.py",                  model_kwargs={'h0': 16, 'h1': 32}),
            DataModel(model_name="model_simple_lin_N_3",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_4",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 16]}),
            DataModel(model_name="model_simple_lin_N_5",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_6",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_7",            model_family=0,     pytorch_file_path="model_02_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 64, 32, 16]}),

            #
            ## Most Basic Perceptron Model. ##
            #
            DataModel(model_name="model_single_linear",             model_family=1,     pytorch_file_path="model_03_single_layer_perceptron.py",       model_kwargs={}),

            #
            ## Mlp flatten first. ##
            #
            DataModel(model_name="model_mlp_flatten_first",         model_family=2,     pytorch_file_path="model_04_mlp_flatten_first.py",             model_kwargs={'h0': 32, 'depth': 2}),

            #
            ## MLP parallel features. ##
            #
            DataModel(model_name="model_parallel_features",         model_family=3,     pytorch_file_path="model_05_parallel_feature_extractors.py",   model_kwargs={'h0': 16, 'depth': 2}),

            #
            ## MLP factorized. ##
            #
            DataModel(model_name="model_factorized",                model_family=4,     pytorch_file_path="model_06_factorized.py",                    model_kwargs={'h0': 8, 'depth': 1}),

            #
            ## MLP Based on global statistics extraction. ##
            #
            DataModel(model_name="model_global_statistics",         model_family=5,     pytorch_file_path="model_07_global_statistics_extractor.py",   model_kwargs={'h0': 16, 'depth': 1}),

            #
            ### ==== Conv Models. ==== ###
            #

            #
            ## Conv1d temporal. ##
            #
            DataModel(model_name="model_conv1d_temporal",           model_family=6,     pytorch_file_path="model_08_conv1d.py",                        model_kwargs={'channels': 8, 'kernel_size': 3, 'pool_size': 2, 'depth': 3}),

            #
            ## Conv1d feature. ##
            #
            DataModel(model_name="model_conv1d_feature",            model_family=7,     pytorch_file_path="model_09_conv1d_features.py",               model_kwargs={'c0': 8, 'k0': 3, 'depth': 1}),

            #
            ## Conv2d standard. ##
            #
            DataModel(model_name="model_conv2d_standard",           model_family=8,     pytorch_file_path="model_10_conv2d_standard.py",               model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 1, 'depth': 3}),

            #
            ## Conv2d depthwise separable. ##
            #
            DataModel(model_name="model_depthwise_separable",       model_family=9,     pytorch_file_path="model_11_conv2d_depthwise_sep.py",          model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 2, 'depth': 1}),

            #
            ## Multiscale CNN. ##
            #
            DataModel(model_name="model_multiscale_cnn",            model_family=10,    pytorch_file_path="model_12_multi_scale_cnn.py",               model_kwargs={'c0': 4, 'depth': 1}),

            #
            ## Stacked Conv2d. ##
            #
            DataModel(model_name="model_stacked_conv2d",            model_family=11,    pytorch_file_path="model_13_stacked_conv2d.py",                model_kwargs={'c0': 8, 'c1': 16, 'k0': 3, 'k1': 3, 'p': 2, 'depth': 2}),

            #
            ## Residual CNN. ##
            #
            DataModel(model_name="model_residual_cnn",              model_family=12,    pytorch_file_path="model_14_residual_cnn.py",                  model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'depth': 2}),

            #
            ### ==== Recurrent Models. ==== ###
            #

            #
            ## Simple RNN. ##
            #
            DataModel(model_name="model_simple_rnn",                model_family=13,    pytorch_file_path="model_15_simple_rnn.py",                    model_kwargs={'h0': 16, 'depth': 1}),

            #
            ## Simple LSTM. ##
            #
            DataModel(model_name="model_lstm",                      model_family=14,    pytorch_file_path="model_16_lstm.py",                          model_kwargs={'h0': 16, 'depth': 1}),

            #
            ## Simple GRU. ##
            #
            DataModel(model_name="model_gru",                       model_family=15,    pytorch_file_path="model_17_gru.py",                           model_kwargs={'h0': 16, 'depth': 1}),

            #
            ## Bidirectional LSTM. ##
            #
            DataModel(model_name="model_bidirectional_lstm",        model_family=16,    pytorch_file_path="model_18_bidirectionnal_lstm.py",           model_kwargs={'h0': 8, 'depth': 1}),

            #
            ## Bidirectional GRU. ##
            #
            DataModel(model_name="model_bidirectional_gru",         model_family=17,    pytorch_file_path="model_19_bidirectionnal_gru.py",            model_kwargs={'h0': 8, 'depth': 1}),

            #
            ### ==== Attention based Models. ==== ###
            #

            #
            ## Self Attention Model. ##
            #
            DataModel(model_name="model_self_attention",            model_family=18,    pytorch_file_path="model_20_self_attention.py",                model_kwargs={'d_k': 16, 'depth': 1}),

            #
            ## Transformer Model. ##
            #
            DataModel(model_name="model_lightweight_transformer",   model_family=19,    pytorch_file_path="model_21_lightweight_transformer.py",       model_kwargs={'d_model': 16, 'num_heads': 1, 'depth': 1}),

            #
            ## Vision Transformer Model. ##
            #
            DataModel(model_name="model_vision_transformer",        model_family=20,    pytorch_file_path="model_22_vision_transformer.py",            model_kwargs={'patch_size': 5, 'd_model': 16, 'num_heads': 2, 'depth': 2, 'mlp_ratio': 4}),

            #
            ### ==== Pooling Models. ==== ###
            #

            #
            ## Global AVG Pooling Model. ##
            #
            DataModel(model_name="model_global_avg_pool",           model_family=21,    pytorch_file_path="model_23_global_avg_pooling.py",            model_kwargs={'h0': 16, 'depth': 2}),

            #
            ## Global Max Pooling Model. ##
            #
            DataModel(model_name="model_global_max_pool",           model_family=22,    pytorch_file_path="model_24_max_pooling.py",                   model_kwargs={'h0': 16, 'depth': 2}),

            #
            ## Global Mixed Pooling Model. ##
            #
            DataModel(model_name="model_mixed_pooling",             model_family=23,    pytorch_file_path="model_25_mixed_pooling.py",                 model_kwargs={'h0': 16, 'depth': 2}),

            #
            ### ==== Hybrid / Specialized Models. ==== ###
            #

            #
            ## CNN / RNN Hybrid. ##
            #
            DataModel(model_name="model_cnn_rnn_hybrid",            model_family=24,    pytorch_file_path="model_26_cnn_rnn_hybrid.py",                model_kwargs={'c0': 8, 'k0': 3, 'h0': 3, 'depth': 1}),

            #
            ## Conv / LSTM hybrid. ##
            #
            DataModel(model_name="model_conv_lstm_hybrid",          model_family=25,    pytorch_file_path="model_27_cnn_lstm_hybrid.py",               model_kwargs={'c0': 8, 'k0': 3, 'h0': 12, 'depth': 1}),

            #
            ## CNN / Attention hybrid. ##
            #
            DataModel(model_name="model_cnn_attention",             model_family=26,    pytorch_file_path="model_28_cnn_attention.py",                 model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'd_k': 12, 'depth': 1}),

            #
            ## TCN Model. ##
            #
            DataModel(model_name="model_tcn",                       model_family=27,    pytorch_file_path="model_29_temporal_cnn.py",                  model_kwargs={'num_channels': 8, 'kernel_size': 3, 'num_layers': 3}),

            #
            ## Senet Model. ##
            #
            DataModel(model_name="model_senet",                     model_family=28,    pytorch_file_path="model_30_senet.py",                         model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4, 'depth': 1}),
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
