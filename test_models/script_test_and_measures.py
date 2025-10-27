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
        pytorch_file_path: str,
        model_kwargs: dict[str, Any],
        main_pt_class_name: str = "Model"
    ) -> None:

        #
        self.model_name: str = model_name
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
            DataModel(model_name="model_simple_lin_1",              pytorch_file_path="model_simple_lin_1.py",                  model_kwargs={'h0': 16}),
            DataModel(model_name="model_simple_lin_2",              pytorch_file_path="model_simple_lin_2.py",                  model_kwargs={'h0': 16, 'h1': 32}),
            DataModel(model_name="model_simple_lin_N_3",            pytorch_file_path="model_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_4",            pytorch_file_path="model_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 16]}),
            DataModel(model_name="model_simple_lin_N_5",            pytorch_file_path="model_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_6",            pytorch_file_path="model_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 32, 16]}),
            DataModel(model_name="model_simple_lin_N_7",            pytorch_file_path="model_simple_lin_N.py",                  model_kwargs={'h_i': [16, 32, 64, 128, 64, 32, 16]}),

            #
            ## Most Basic Perceptron Model. ##
            #
            DataModel(model_name="model_single_linear",             pytorch_file_path="model_single_layer_perceptron.py",       model_kwargs={}),

            #
            ## Mlp flatten first. ##
            #
            DataModel(model_name="model_mlp_flatten_first",         pytorch_file_path="model_mlp_flatten_first.py",             model_kwargs={'h0': 32}),

            #
            ## MLP parallel features. ##
            #
            DataModel(model_name="model_parallel_features",         pytorch_file_path="model_parallel_feature_extractors.py",   model_kwargs={'h0': 16}),

            #
            ## MLP factorized. ##
            #
            DataModel(model_name="model_factorized",                pytorch_file_path="model_factorized.py",                    model_kwargs={'h0': 8}),

            #
            ## MLP Based on global statistics extraction. ##
            #
            DataModel(model_name="model_global_statistics",         pytorch_file_path="model_global_statistics_extractor.py",   model_kwargs={'h0': 16}),

            #
            ### ==== Conv Models. ==== ###
            #

            #
            ## Conv1d temporal. ##
            #
            DataModel(model_name="model_conv1d_temporal",           pytorch_file_path="model_conv1d.py",                        model_kwargs={'c0': 8, 'k0': 3, 'p0': 2}),

            #
            ## Conv1d feature. ##
            #
            DataModel(model_name="model_conv1d_feature",            pytorch_file_path="model_conv1d_features.py",               model_kwargs={'c0': 8, 'k0': 3}),

            #
            ## Conv2d standard. ##
            #
            DataModel(model_name="model_conv2d_standard",           pytorch_file_path="model_conv2d_standard.py",               model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 2}),

            #
            ## Conv2d depthwise separable. ##
            #
            DataModel(model_name="model_depthwise_separable",       pytorch_file_path="model_conv2d_depthwise_sep.py",          model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'p': 2}),

            #
            ## Multiscale CNN. ##
            #
            DataModel(model_name="model_multiscale_cnn",            pytorch_file_path="model_multi_scale_cnn.py",               model_kwargs={'c0': 4}),

            #
            ## Stacked Conv2d. ##
            #
            DataModel(model_name="model_stacked_conv2d",            pytorch_file_path="model_stacked_conv2d.py",                model_kwargs={'c0': 8, 'c1': 16, 'k0': 3, 'k1': 3, 'p': 2}),

            #
            ## Residual CNN. ##
            #
            DataModel(model_name="model_residual_cnn",              pytorch_file_path="model_residual_cnn.py",                  model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3}),

            #
            ### ==== Recurrent Models. ==== ###
            #

            #
            ## Simple RNN. ##
            #
            DataModel(model_name="model_simple_rnn",                pytorch_file_path="model_simple_rnn.py",                    model_kwargs={'h0': 16}),

            #
            ## Simple LSTM. ##
            #
            DataModel(model_name="model_lstm",                      pytorch_file_path="model_lstm.py",                          model_kwargs={'h0': 16}),

            #
            ## Simple GRU. ##
            #
            DataModel(model_name="model_gru",                       pytorch_file_path="model_gru.py",                           model_kwargs={'h0': 16}),

            #
            ## Bidirectional LSTM. ##
            #
            DataModel(model_name="model_bidirectional_lstm",        pytorch_file_path="model_bidirectionnal_lstm.py",           model_kwargs={'h0': 8}),

            #
            ## Bidirectional GRU. ##
            #
            DataModel(model_name="model_bidirectional_gru",         pytorch_file_path="model_bidirectionnal_gru.py",            model_kwargs={'h0': 8}),

            #
            ### ==== Attention based Models. ==== ###
            #

            #
            ## Self Attention Model. ##
            #
            DataModel(model_name="model_self_attention",            pytorch_file_path="model_self_attention.py",                model_kwargs={'d_k': 16}),

            #
            ## Transformer Model. ##
            #
            DataModel(model_name="model_lightweight_transformer",   pytorch_file_path="model_lightweight_transformer.py",       model_kwargs={'d_model': 16, 'num_heads': 1, 'depth': 1}),

            #
            ## Vision Transformer Model. ##
            #
            DataModel(model_name="model_vision_transformer",        pytorch_file_path="model_vision_transformer.py",            model_kwargs={'patch_size': 5, 'd_model': 16, 'num_heads': 2, 'depth': 2, 'mlp_ratio': 4}),

            #
            ### ==== Pooling Models. ==== ###
            #

            #
            ## Global AVG Pooling Model. ##
            #
            DataModel(model_name="model_global_avg_pool",           pytorch_file_path="model_global_avg_pooling.py",            model_kwargs={'h0': 16}),

            #
            ## Global Max Pooling Model. ##
            #
            DataModel(model_name="model_global_max_pool",           pytorch_file_path="model_max_pooling.py",                   model_kwargs={'h0': 16}),

            #
            ## Global Mixed Pooling Model. ##
            #
            DataModel(model_name="model_mixed_pooling",             pytorch_file_path="model_mixed_pooling.py",                 model_kwargs={'h0': 16}),

            #
            ### ==== Hybrid / Specialized Models. ==== ###
            #

            #
            ## CNN / RNN Hybrid. ##
            #
            DataModel(model_name="model_cnn_rnn_hybrid",            pytorch_file_path="model_cnn_rnn_hybrid.py",                model_kwargs={'c0': 8, 'k0': 3, 'h0': 3}),

            #
            ## Conv / LSTM hybrid. ##
            #
            DataModel(model_name="model_conv_lstm_hybrid",          pytorch_file_path="model_cnn_lstm_hybrid.py",               model_kwargs={'c0': 8, 'k0': 3, 'h0': 12}),

            #
            ## CNN / Attention hybrid. ##
            #
            DataModel(model_name="model_cnn_attention",             pytorch_file_path="model_cnn_attention.py",                 model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'd_k': 12}),

            #
            ## TCN Model. ##
            #
            DataModel(model_name="model_tcn",                       pytorch_file_path="model_temporal_cnn.py",                  model_kwargs={'num_channels': 8, 'kernel_size': 3, 'num_layers': 3}),

            #
            ## Senet Model. ##
            #
            DataModel(model_name="model_senet",                     pytorch_file_path="model_senet.py",                         model_kwargs={'c0': 8, 'k_h': 3, 'k_w': 3, 'reduction_ratio': 4}),
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
