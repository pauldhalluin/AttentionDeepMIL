from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags
from model import Attention, GatedAttention
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Test settings
parser = argparse.ArgumentParser(description='Test ')

parser.add_argument('--feature_path', type=str, default='', help='path for features')
parser.add_argument('--model_path', type=str, default='', help='path for model')
parser.add_argument('--csv_path', type=str, default='', help='path for predictions')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)

def get_X_test(args, list_samples):

    X_test = np.zeros((len(list_samples), 1000, 2048))

    for i, sample in enumerate(list_samples):
        X_test[i] = np.load(os.path.join(args.feature_path, sample))[:, 3:]

    X_test = torch.Tensor(X_test)

    return X_test


def eval(X_test):
    model.eval()
    list_pred = []
    for sample in X_test:
        Y_prob, _, _ = model(torch.unsqueeze(sample, dim=0))
        list_pred.append(Y_prob.detach().numpy()[0, 0])
    return np.array(list_pred)


if __name__ == "__main__":

    list_samples = os.listdir(args.feature_path)
    list_models = os.listdir(args.model_path)

    preds = np.zeros((len(list_models), len(list_samples)))

    for i, model_dict in enumerate(list_models):
        if args.model=='attention':
            model = Attention()
        elif args.model=='gated_attention':
            model = GatedAttention()
        checkpoint = torch.load(os.path.join(args.model_path, model_dict))
        model.load_state_dict(checkpoint)
        X_test = get_X_test(args, list_samples)
        array_pred = eval(X_test)
        preds[i] = array_pred

    preds = np.mean(preds, axis=0)

    submission = pd.DataFrame(
        {"Sample ID": list_samples, "Target": preds}
    ).sort_values(
        "Sample ID"
    )  # extra step to sort the sample IDs

    # sanity checks
    assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
    assert submission.shape == (149, 2), "Your submission file must be of shape (149, 2)"
    assert list(submission.columns) == [
        "Sample ID",
        "Target",
    ], "Your submission file must have columns `Sample ID` and `Target`"

    # save the submission as a csv file
    submission.to_csv(os.path.join(args.csv_path, "test_output.csv", index=None))