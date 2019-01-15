from hw3_utils import *
import math


def euclidean_distance(features_1, features_2):
    count = 0
    for feature_1, feature_2 in zip(features_1, features_2):
        count += (feature_1 - feature_2) ** 2
    return math.sqrt(count)


class knn_classifier(abstract_classifier):
    def __init__(self, data, labels, k=10):
        self.data = data
        self.labels = labels
        self.k = k

    def classify(self, features):
        nn_labels = {True: 0, False: 0}
        distances = []
        for known_features in self.data:
            distances.append(euclidean_distance(known_features, features))
        for i in range(self.k):
            nn_idx = distances.index(min(distances))
            nn_labels[self.labels[nn_idx]] += 1
            distances[nn_idx] = float('inf')
        if nn_labels[True] > nn_labels[False]:
            return 1
        else:
            return 0


class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(data, labels, self.k)


train_features, train_labels, test_features = load_data()
factory = knn_factory()
classifier = factory.train(train_features, train_labels)

success = 0.0
for i in range(len(train_features)):
    if not i % 50:
        print(i)
    if classifier.classify(train_features[i]) == train_labels[i]:
        success += 1
print(success)

