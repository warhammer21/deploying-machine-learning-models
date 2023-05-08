import pandas as pd
import numpy as np
import yaml
import re

config_path = '/Users/shreyakvashisht/PycharmProjects/deploying-machine-learning-models/titanic/classification_model/config.yml'
def find_and_read_config():
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
config = find_and_read_config()
# print(config['data_file'])
