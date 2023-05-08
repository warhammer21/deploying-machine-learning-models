titanic_price = '/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/trained_models/pipeline.pkl'
input_data_path = '/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/datasets/X_test.csv'
import joblib
import pandas as pd
import typing as t

def make_predictions(input_data: t.Union[pd.DataFrame, dict],read_from_path = True):
    pipeline = joblib.load(titanic_price)
    if read_from_path==True:
        input_data = pd.read_csv(input_data_path)
    else:
         input_data = pd.DataFrame(input_data)
    result = pipeline.predict(input_data)
    return result

test = make_predictions({'asa':'dsds'},True)
print(test)
