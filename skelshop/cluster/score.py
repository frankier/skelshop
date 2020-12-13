from operator import itemgetter


def silhouette_scorer(estimator, X):
    from sklearn import metrics

    estimator.fit(X)
    cluster_labels = estimator.labels_
    seen_labels = set()
    for label in cluster_labels:
        seen_labels.add(label)
        if len(seen_labels) > 1:
            return metrics.silhouette_score(X, cluster_labels, metric="cosine")
    return 0


# def macc(tp, tn, fp, fn):
# denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
# if denom == 0:
# return 0
# return (tp * tn - fp * fn) / denom ** 0.5


def acc(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def tracks_macc(estimator, X, y):
    from itertools import groupby

    estimator.fit(X)
    cluster_labels = estimator.labels_
    idx = 0
    tp = tn = fp = fn = 0
    for grp, items in groupby(y, itemgetter(0, 1)):
        pers_ids = [pers_id for _, _, pers_id in items]
        end_idx = idx + len(pers_ids)
        label_slice = cluster_labels[idx:end_idx]
        for pers_id1, label1 in zip(pers_ids, label_slice):
            for pers_id2, label2 in zip(pers_ids, label_slice):
                if pers_id1 == pers_id2:
                    if label1 != -1 and label1 == label2:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if label1 == -1 or label1 == label2:
                        fp += 1
                    else:
                        tn += 1
        idx = end_idx
    score = acc(tp, tn, fp, fn)
    return score
