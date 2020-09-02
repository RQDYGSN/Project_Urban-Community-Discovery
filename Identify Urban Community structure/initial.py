# import numpy as np
import pandas as pd

def initial():
    adj = pd.read_csv('./input/adj.csv', header=None).values
    features = pd.read_csv('./input/features.csv', header=None).values
    threshold = pd.read_csv('./input/threshold.csv', header=None).values[0]
    return adj, features, threshold
