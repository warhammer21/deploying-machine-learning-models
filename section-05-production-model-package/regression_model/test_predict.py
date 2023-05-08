from regression_model.config.core import config
from regression_model.trained_models.predict import  make_prediction
from regression_model.processing.data_manager import load_pipeline
from regression_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def test_p(sample_input_data):
    results = make_prediction(input_data=sample_input_data)
    print(test_p)

test_p(sample_input_data)

