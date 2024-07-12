import pickle # https://www.geeksforgeeks.org/understanding-python-pickling-example/
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import - see import from data.py
from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array (features?; contains training data)
        Training data.
    y_train : np.array  (contains the corresponding lables for the data)
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
   # TODO: implement the function 
   # Initializes the Logic Regression Model using scikit; need to import
    model = LogisticRegression()

   # Train model on training data
   # https://scipy-lectures.org/packages/scikit-learn/#:~:text=In%20all%20Estimators%3A-,model.,the%20data%20X%20(e.g.%20model.
   # https://stackoverflow.com/questions/62566791/keras-how-to-properly-use-fit-to-train-a-model
   # fit, adjusts the model parameters to min error bw predictions/labels
    model.fit(X_train, y_train)

   # Return the trained model
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    #sklearn metrics
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X): #defines function named inference w 2 arguments (inputs)
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ??? #Can be any model that has a 'predict' method
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: implement the function
    # https://stackoverflow.com/questions/72754270/how-to-use-predict-in-a-linear-regression-model
    preds = model.predict(X)
    
    return preds

def save_model(model, path):
    """ Serializes (saves) model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder. (trained?)
    path : str
        Path to save pickle file.
    """
    # TODO: implement the function
    # open file
    with open(path, 'wb') as file: # write
        # https://www.datacamp.com/tutorial/pickle-python-tutorial
        # https://www.youtube.com/watch?v=6Q56r_fVqgw
        pickle.dump(model, file) # searilizes (obj to be dumped, to this file)


def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # TODO: implement the function
    # https://www.digitalocean.com/community/tutorials/python-pickle-example
    # https://www.geeksforgeeks.org/understanding-python-pickling-example/
    with open(path, 'rb') as file: # read
        return pickle.load(file) # (unpickle this file)

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
): # function computes model performance metrics on a specific slice of data
   # defined by a col name and specific value in that col 
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    process the data slice
    make predictions using the model
    compute model performance metrics on the slice

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
        ---------------
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # TODO: implement the function
    # Filter df to include only rows where 'column_name' == 'slice_value'
    data_slice = data[data[column_name] == slice_value]

    # Process data; ml.data process_data
    # X = categorical, y = labels
    # https://rdrr.io/github/jeremyrcoyle/sl3/man/process_data.html

    X_slice, y_slice, _, _ = process_data(
        # your code here
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False; if false it indicates use for inference or validation
        # label=label, etc ? see train_model.py,label="salary"
        data_slice,
        categorical_features=categorical_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        training=False
    )

    # Make predictions
    # Directions: use inference function on X_slice
    preds = inference(model, X_slice) # your code here to get prediction on X_slice using the inference function
    
    # Metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
