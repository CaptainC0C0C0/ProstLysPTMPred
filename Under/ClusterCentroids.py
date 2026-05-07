from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter


def UnderClusterCentroids(request, X, label):
    n_init = int(request.form.get(f'CCn_init'))
    voting = request.form.get(f'CC_voting')

    label_counts = Counter(label.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'ClusterCentroids{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    x_resampled, y_resampled = Cluster_Centroids(X, label, label_count_dict, n_init, voting)
    return x_resampled, y_resampled


def Cluster_Centroids(X, y, sampling_strategy, n_init, voting):
    y = y.ravel()
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy, random_state=1, voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=1, batch_size=2048)
    )
    x_resampled, y_resampled = under.fit_resample(X, y)
    return x_resampled, y_resampled