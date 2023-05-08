import pandas as pd
import numpy as np
from config.core import config,get_first_cabin,get_title
from sklearn.model_selection import train_test_split
from pipeline import titanic_pipe
import joblib

from processing import features as ff
def run_training():
    data = pd.read_csv(config['data_file'])
    # replace interrogation marks by NaN values
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')
    data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config['target'], axis=1),  # predictors
        data[config['target']],  # target
        test_size=config['test_size'],  # percentage of obs in test set
        random_state=config['random_state'])
    X_test.to_csv('/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/datasets/X_test.csv',
                  index = False)# seed to ensure reproducibility


    titanic_pipe.fit(X_train, y_train)
    joblib.dump(titanic_pipe, '/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/trained_models/pipeline.pkl')
    print('worked')

    data = pd.read_csv(config['data_file'])
    # replace interrogation marks by NaN values
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].apply(get_first_cabin)
    data['title'] = data['name'].apply(get_title)
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')
    data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config['target'], axis=1),  # predictors
        data[config['target']],  # target
        test_size=config['test_size'],  # percentage of obs in test set
        random_state=config['random_state'])  # seed to ensure reproducibility

    titanic_pipe.fit(X_train, y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(titanic_pipe.predict(X_test))
    joblib.dump(titanic_pipe, '/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/trained_models/pipeline.pkl')
    print('worked')
if __name__ == "__main__":
    run_training()
