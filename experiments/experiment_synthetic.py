import numpy as np
import random
import torch

import os, sys
sys.path.insert(0, os.path.abspath(".."))
from datasets import datasets
from cqr_comparison import ConformalizedQR, RandomForestQR, NeuralNetworkQR

import pdb

verbose = True

n_test = 20000

def generate_data(n):
    def f(Z):
        return(2.0*np.sin(np.pi*Z) + np.pi*Z)
    p = 100
    X = np.random.uniform(size=(n,p))
    beta = np.zeros((p,))
    beta[0:5] = 1.0
    Z = np.dot(X,beta)
    E = np.random.normal(size=(n,))
    Y = f(Z) + np.sqrt(1.0+Z**2) * E
    return X.astype(np.float32), Y.astype(np.float32)

def experiment(params):
    """ Compute prediction intervals on synthetic data.
    Print average length and coverage.

    Parameters
    ----------
    params : a dictionary with the following fields
       'n'            : number of training samples
       'seed'         : integer, random random_state to be used

    """

    # Extract main parameters
    n_train = int(params["n"])
    seed = int(params["seed"])

    # Name of output file
    dataset_name = "synthetic"
    params["method"] = "cqr_quantile_net"
    ratio_train = 75
    outfile = dataset_name + "/" + params["method"] + "_n" + str(n_train) + "_seed" + str(seed)

    # Determines the size of test set
    test_ratio = 0.2

    # Alpha for conformal prediction intervals
    significance = 0.1

    # Quantiles for neural network
    quantiles_net = [0.1, 0.5, 0.9]

    # List of conformalization methods
    conf_methods_list = ["CQR", "CQRm", "CQRr"]

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ## Generate the data
    X_train, y_train = generate_data(n_train)
    X_test, y_test = generate_data(n_test)

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

    if params["method"] == 'cqr_quantile_net':
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
    else:
        print("Uknown method.")
        sys.exit()

    cqr = ConformalizedQR(model, model, X_train, y_train, idx_train, idx_cal, significance)

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
    params['n']    = 1000    # Number of training observarions
    params['seed'] = 123     # Seed for pseudo-random numbers

    # Run experiment
    experiment(params)
