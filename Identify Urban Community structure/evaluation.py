import numpy as np


def nmi(partition, ground_truth):
    node_num = 0
    for k, v in partition.items():
        node_num += len(v)

    hx = 0
    for k, v in partition.items():
        p = len(v)/node_num
        hx += p*np.log2(1/(p+0.0000001))

    hy = 0
    for k, v in ground_truth.items():
        p = len(v)/node_num
        hy += p*np.log2(1/(p+0.0000001))

    ixy = 0
    for k1, v1 in partition.items():
        for k2, v2 in ground_truth.items():
            pxy = len(set(v1) & set(v2))/node_num
            px = len(v1)/node_num
            py = len(v2)/node_num
            ixy += pxy*np.log2((pxy+0.0000001)/(px+0.0000001)/(py+0.0000001))

    return ixy*2/(hx+hy)


def fscore(partition, ground_truth):
    F = 0
    count = 0
    for k1, v1 in partition.items():
        f = 0
        s1 = set(v1)
        for k2, v2 in ground_truth.items():
            s2 = set(v2)
            intersection = s1 & s2
            precision = len(intersection)/len(s1)
            recall = len(intersection)/len(s2)
            if precision*recall != 0:
                tmp = 2*precision*recall/(precision + recall)
                if tmp > f: f = tmp
            else:
                continue
        F += f
        count += 1
    F = F/count
    return F


def jaccard_similarity(partition, ground_truth):
    JS = 0
    count = 0
    for k1, v1 in partition.items():
        js = 0
        s1 = set(v1)
        for k2, v2 in ground_truth.items():
            s2 = set(v2)
            tmp = len(s1 & s2)/len(s1 | s2)
            if tmp>js: js = tmp
        JS += js
        count += 1
    JS = JS/count
    return JS