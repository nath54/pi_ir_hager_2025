#
### Import Modules. ###
#
from typing import Any
#
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
        model_args: list[Any],
        model_kwargs: dict[str, Any],
        main_pt_class_name: str = "Model"
    ) -> None:

        #
        self.model_name: str = model_name
        self.pytorch_file_path: str = pytorch_file_path
        self.main_pt_class_name: str = main_pt_class_name
        #
        self.model_args: list[Any] = model_args
        self.model_kwargs: dict[str, Any] = model_kwargs


#
### Data models Class. ###
#
class DataModelsClass:

    #
    def __init__(self) -> None:

        #
        self.models_to_test: list[DataModel] = [

            DataModel(model_name="model_simple_lin_1",              pytorch_file_path="model_simple_lin_1.py",                  model_args=[], model_kwargs={}),
            DataModel(model_name="model_simple_lin_2",              pytorch_file_path="model_simple_lin_2.py",                  model_args=[], model_kwargs={}),
            DataModel(model_name="model_simple_lin_N",              pytorch_file_path="model_simple_lin_N.py",                  model_args=[], model_kwargs={}),
            DataModel(model_name="model_single_linear",             pytorch_file_path="model_single_layer_perceptron.py",       model_args=[], model_kwargs={}),
            DataModel(model_name="model_mlp_flatten_first",         pytorch_file_path="model_mlp_flatten_first.py",             model_args=[], model_kwargs={}),
            DataModel(model_name="model_parallel_features",         pytorch_file_path="model_parallel_feature_extractors.py",   model_args=[], model_kwargs={}),
            DataModel(model_name="model_factorized",                pytorch_file_path="model_factorized.py",                    model_args=[], model_kwargs={}),
            DataModel(model_name="model_global_statistics",         pytorch_file_path="model_global_statistics_extractor.py",   model_args=[], model_kwargs={}),

            DataModel(model_name="model_conv1d_temporal",           pytorch_file_path="model_conv1d.py",                        model_args=[], model_kwargs={}),
            DataModel(model_name="model_conv1d_feature",            pytorch_file_path="model_conv1d_features.py",               model_args=[], model_kwargs={}),
            DataModel(model_name="model_conv2d_standard",           pytorch_file_path="model_conv2d_standard.py",               model_args=[], model_kwargs={}),
            DataModel(model_name="model_depthwise_separable",       pytorch_file_path="model_conv2d_depthwise_sep.py",          model_args=[], model_kwargs={}),
            DataModel(model_name="model_multiscale_cnn",            pytorch_file_path="model_multi_scale_cnn.py",               model_args=[], model_kwargs={}),
            DataModel(model_name="model_stacked_conv2d",            pytorch_file_path="model_stacked_conv2d.py",                model_args=[], model_kwargs={}),
            DataModel(model_name="model_residual_cnn",              pytorch_file_path="model_residual_cnn.py",                  model_args=[], model_kwargs={}),

            DataModel(model_name="model_simple_rnn",                pytorch_file_path="model_simple_rnn.py",                    model_args=[], model_kwargs={}),
            DataModel(model_name="model_lstm",                      pytorch_file_path="model_lstm.py",                          model_args=[], model_kwargs={}),
            DataModel(model_name="model_gru",                       pytorch_file_path="model_gru.py",                           model_args=[], model_kwargs={}),
            DataModel(model_name="model_bidirectional_lstm",        pytorch_file_path="model_bidirectionnal_lstm.py",           model_args=[], model_kwargs={}),
            DataModel(model_name="model_bidirectional_gru",         pytorch_file_path="model_bidirectionnal_gru.py",            model_args=[], model_kwargs={}),

            DataModel(model_name="model_self_attention",            pytorch_file_path="model_self_attention.py",                model_args=[], model_kwargs={}),
            DataModel(model_name="model_lightweight_transformer",   pytorch_file_path="model_lightweight_transformer.py",       model_args=[], model_kwargs={}),

            DataModel(model_name="model_global_avg_pool",           pytorch_file_path="model_global_avg_pooling.py",            model_args=[], model_kwargs={}),
            DataModel(model_name="model_global_max_pool",           pytorch_file_path="model_max_pooling.py",                   model_args=[], model_kwargs={}),
            DataModel(model_name="model_mixed_pooling",             pytorch_file_path="model_mixed_pooling.py",                 model_args=[], model_kwargs={}),

            DataModel(model_name="model_cnn_rnn_hybrid",            pytorch_file_path="model_cnn_rnn_hybrid.py",                model_args=[], model_kwargs={}),
            DataModel(model_name="model_conv_lstm_hybrid",          pytorch_file_path="model_cnn_lstm_hybrid.py",               model_args=[], model_kwargs={}),
            DataModel(model_name="model_cnn_attention",             pytorch_file_path="model_cnn_attention.py",                 model_args=[], model_kwargs={}),
            DataModel(model_name="model_tcn",                       pytorch_file_path="model_temporal_cnn.py",                  model_args=[], model_kwargs={}),
            DataModel(model_name="model_senet",                     pytorch_file_path="model_senet.py",                         model_args=[], model_kwargs={}),
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
        return getattr(obj, model_data.main_pt_class_name)(*model_data.model_args, **model_data.model_kwargs)


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
        pt_model: nn.Module = self.data_models.load_model(model_idx=model_idx)

        #
        self.mpt.measure_model(
            model_name=self.data_models.models_to_test[model_idx].model_name,
            pt_model=pt_model
        )


    #
    def main(self) -> None:

        #
        for model_idx in range(len(self.data_models.models_to_test)):

            #
            self.test_one_model(model_idx=model_idx)

        #
        self.mpt.save_logs()


#
if __name__ == "__main__":

    #
    main: MainTestAndMeasures = MainTestAndMeasures()

    #
    main.main()
