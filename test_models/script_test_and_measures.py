#
### Import Modules. ###
#
from typing import Any
#
from torch import nn
#
from lib_import_pt_models import import_module_from_filepath


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
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
            DataModel(model_name="", pytorch_file_path="", model_args=[], model_kwargs={}),
        ]

    #
    def load_model(self, model_idx: int) -> nn.Module:

        #
        model_data: DataModel = self.models_to_test[model_idx]

        #
        obj: object = import_module_from_filepath(filepath=model_data.pytorch_file_path)

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
    def main(self) -> None:

        #
        pass


#
if __name__ == "__main__":

    #
    main: MainTestAndMeasures = MainTestAndMeasures()

    #
    main.main()
