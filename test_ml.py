import pytest
import pandas as pd
import numpy as np
import os
from ml import data, model
from sklearn.linear_model import LogisticRegression
# TODO: add necessary import
# https://medium.com/@ydmarinb/simplifying-unit-testing-in-machine-learning-with-python-df9b9c1a3300
# https://eugeneyan.com/writing/unit-testing-ml/

@pytest.fixture(scope="session") #see train_model.py
def test_data_path():
    # Construct the path to your test data
    test_path = os.getcwd()
    test_data_path = os.path.join(test_path, "data", "census.csv")
    df = pd.read_csv(test_data_path) # your code here
    return df

# TODO: implement the first test. Change the function name and input as needed
# Return the expected type of result (isinstance); isinstance(obj, classinfo)
# https://www.w3schools.com/python/ref_func_isinstance.asp

def test_train_model():
    """
    # Return the expected type of result
    """
    # Your code here
    X_train = np.random.rand(100, 25)
    y_train = np.random.randint(2, size=100)

    #Training the model
    model_output = model.train_model(X_train, y_train)

    #Check model type
    assert isinstance(model_output, LogisticRegression)


# TODO: implement the second test. Change the function name and input as needed
# keep it SIMPLE!
def test_dataset_length(test_data_path):
    """
    # Test to make sure the dataset has enough info
    """
    # Your code here
    assert test_data_path.shape[0] > 10000
    


# TODO: implement the third test. Change the function name and input as needed
# keep it SIMPLE!
def test_dataset_col(test_data_path):
    """
    # Test to make sure there are 15 cols
    """
    # Your code here
    assert test_data_path.shape[1] == 15
