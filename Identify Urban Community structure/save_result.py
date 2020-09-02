import main

import numpy as np
import pandas as pd

# save results
def save_result(partition_fast):
    labels = np.zeros(main.node_num)
    for k, v in partition_fast.items():
        for i in v:
            labels[i] = k
    pd.DataFrame(labels, index=None, columns=None).to_csv('./output/identified_labels.csv')