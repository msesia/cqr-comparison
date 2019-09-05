import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os, sys
sys.path.insert(0, os.path.abspath(".."))
from datasets import datasets
from cqr_comparison import ConformalizedQR, RandomForestQR, NeuralNetworkQR

import pdb

base_dataset_path = '../datasets/'

verbose = True

def experiment(params):
    """ Estimate prediction intervals and print the average length and coverage

    Parameters
    ----------
    params : a dictionary with the following fields
       'dataset_name' : string, name of dataset
       'method'       : string, conformalization method
       'level'        : string, nominal level for black-box (either 'fixed' or 'cv')
       'ratio'        : numeric, percentage of data used for training
       'seed'         : random seed
    """

    # Extract main parameters
    dataset_name = params["data"]
    method = params["method"]
    level = params["level"]
    ratio_train = params["ratio"]
    seed = params["seed"]

    # Determines the size of test set
    test_ratio = 0.2

    # conformal prediction miscoverage level
    significance = 0.1

    # Quantiles
    quantiles = [0.05, 0.95]

    # Quantiles for training
    if level == "cv":
        quantiles_net = [0.1, 0.5, 0.9]
    else:
        quantiles_net = [0.05, 0.5, 0.95]

    # List of conformalization methods
    conf_methods_list = ["CQR", "CQRm", "CQRr"]

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load the data
    try:
        X, y = datasets.GetDataset(dataset_name, base_dataset_path)
        print("Loaded dataset '" + dataset_name + "'.")
        sys.stdout.flush()
    except:
        print("Error: cannot load dataset " + dataset_name)
        return

    # Dataset is divided into test and train data based on test_ratio parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    # Reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    n_train = X_train.shape[0]

    # Print input dimensions
    print("Data size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1],
                                                        X_test.shape[0], X_test.shape[1]))
    sys.stdout.flush()

    # Set seed for splitting the data into proper train and calibration
    np.random.seed(seed)
    idx = np.random.permutation(n_train)

    # Divide the data into proper training set and calibration set
    n_half = int(np.floor(n_train * ratio_train / 100.0))
    idx_train, idx_cal = idx[:n_half], idx[n_half:n_train]

    # Zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)

    # Scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train)/mean_ytrain
    y_test = np.squeeze(y_test)/mean_ytrain

    if params["method"] == 'cqr_quantile_forest':
        # Parameters of the random forest
        params = dict()
        params['n_estimators'] = 1000
        params['max_features'] = X_train.shape[1]
        params['min_samples_leaf'] = 1
        params['random_state'] = seed
        params['n_jobs'] = 5
        params['cv'] = (level=="cv")
        # Initialize random forest regressor
        model = RandomForestQR(params, quantiles, verbose=verbose)
        # Initialize regressor for hyperparameter tuning
        model_tuning = RandomForestQR(params, quantiles, verbose=verbose)

    elif params["method"] == 'cqr_quantile_net':
        # Parameters of the neural network
        params = dict()
        params['in_shape'] = X_train.shape[1]
        params['epochs'] = 1000
        params['lr'] = 0.0005
        params['hidden_size'] = 64
        params['batch_size'] = 64
        params['dropout'] = 0.1
        params['wd'] = 1e-6
        params['test_ratio'] = 0.05
        params['random_state'] = seed
        # Initialize neural network regressor
        model = NeuralNetworkQR(params, quantiles_net, verbose=verbose)
        # Initialize regressor for hyperparameter tuning
        model_tuning = NeuralNetworkQR(params, quantiles, verbose=verbose)

    else:
        print("Uknown method.")
        sys.exit()

    cqr = ConformalizedQR(model, model_tuning, X_train, y_train, idx_train, idx_cal, significance)

    for conf_method in conf_methods_list:
        # Compute CQR intervals
        lower, upper = cqr.predict(X_test, y_test, significance, method = conf_method)

        # Compute coverage and widths
        covered = (y_test >= lower) & (y_test <= upper)
        widths = upper-lower

        # Print update
        print(conf_method + ": " + "coverage %.3f, width %.3f" %(np.mean(covered), np.mean(widths)))
        sys.stdout.flush()

if __name__ == '__main__':
    # Parameters for this experiment
    params = dict()
    params['data']   = "star"                # Name of dataset
    params['method'] = "cqr_quantile_net"    # Black-box method ("cqr_quantile_net" or "cqr_quantile_forest")
    params['level']  = "fixed"               # Whether to tune the black-box ("fixed" or "cv")
    params['ratio']  = 50                    # Percentage of data used for training
    params['seed']   = 1                     # Random seed

    # Run experiment
    experiment(params)
