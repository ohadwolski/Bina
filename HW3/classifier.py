from hw3_utils import *
import math


def euclidean_distance(features_1, features_2):
    return np.linalg.norm(features_1 - features_2) ## REMOVE BEFORE SUBMISSSION
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
        distances = []
        for known_features in self.data:
            distances.append(euclidean_distance(known_features, features))

        votes = {True: 0, False: 0}
        for _ in range(self.k):
            nn_idx = distances.index(min(distances))
            votes[self.labels[nn_idx]] += 1
            distances[nn_idx] = float('inf')

        if votes[True] > votes[False]:
            return 1
        else:
            return 0


class knn_factory(abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        return knn_classifier(data, labels, self.k)


import random
import pickle
FOLD_FILENAME = "ecg_fold_%d.data"


def split_crosscheck_groups(dataset, num_folds):
    features = dataset[0]
    labels = dataset[1]

    tags = [False, True]
    size_added = {tag: 0 for tag in tags}
    label_idx = {tag: [] for tag in tags}
    for i, label in enumerate(labels):
        label_idx[label].append(i)
    for tag in tags:
        random.shuffle(label_idx[tag])

    folds = []
    for i in range(num_folds):
        fold_idxs = []
        for tag in tags:
            if i == num_folds - 1:
                fold_tag_size = len(label_idx[tag]) - size_added[tag]
            elif tag == tags[-1]:
                fold_tag_size = math.floor(len(labels) / num_folds) - len(fold_idxs)
            else:
                fold_tag_size = math.floor(len(label_idx[tag]) / num_folds)
            fold_tag_idx = label_idx[tag][size_added[tag]:size_added[tag] + fold_tag_size]
            size_added[tag] += fold_tag_size
            fold_idxs += fold_tag_idx
        random.shuffle(fold_idxs)
        new_fold_labels = []
        new_fold_features = []
        for fold_idx in fold_idxs:
            new_fold_features.append(features[fold_idx])
            new_fold_labels.append(labels[fold_idx])
        folds.append((new_fold_features, new_fold_labels))

    for i, fold in enumerate(folds):
        with open(FOLD_FILENAME % (i+1), 'wb') as fd:
            pickle.dump(fold, fd)


def load_k_fold_data(fold_idx):
    with open(FOLD_FILENAME % fold_idx, 'rb') as fd:
        return pickle.load(fd)


def evaluate(classifier_factory, k):
    success_count = 0.0
    total_count = 0.0
    success = {True: 0, False: 0}
    failure = {True: 0, False: 0}
    for i in range(k):
        test = load_k_fold_data(i+1)
        train_features = []
        train_labels = []
        for j in range(k):
            if i == j:
                continue
            train = load_k_fold_data(j+1)
            train_features += train[0]
            train_labels += train[1]
        train_labels = np.array(train_labels)
        train_features = np.array(train_features)
        test_labels = np.array(test[1])
        test_features = np.array(test[0])
        classifier = classifier_factory.train(train_features, train_labels)
        for features, label in zip(test_features, test_labels):
            inference_results = classifier.classify(features)
            success_count += inference_results == label
            if inference_results == label:
                success[label] += 1
            else:
                failure[label] += 1
            total_count += 1
    accuracy = success_count / total_count
    error = 1.0 - accuracy
    print("True Positive=%d" % success[True], "True Positive=%d" % success[False])
    print("False Positive=%d" % failure[True], "False Negative=%d" % failure[False])
    return accuracy, error


# Load Features
train_features, train_labels, test_features = load_data()

# 3.2
num_folds = 10
split_crosscheck_groups((train_features, train_labels), num_folds)


# 3.5
accuracy = {}
error = {}
for k in [1, 3, 5, 7, 13]:
    print("K=%d" % k)
    knn = knn_factory(k)
    accuracy[k], error[k] = evaluate(knn, num_folds)
    print("K=%d:" % k, "Accuracy=%f" % accuracy[k], "Error=%f" % error[k])
    print()

# 3.5.1
import csv
with open("experiments6.csv", 'w') as fd:
    csv_writer = csv.writer(fd)
    for k in accuracy:
        csv_writer.writerow([k, accuracy[k], error[k]])

# 3.5.2
import matplotlib.pyplot as plt
plt.figure()
plt.title("Accuracy as function of K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(accuracy.keys(), accuracy.values())
plt.show()


# 7
import sklearn.tree
import sklearn.linear_model


class tree_classifier(abstract_classifier):
    def __init__(self, tree):
        self.tree = tree

    def classify(self, features):
        return self.tree.predict(features.reshape(1, -1))[0]


class tree_factory(abstract_classifier_factory):
    name = "Decision Tree"

    def __init__(self):
        pass

    def train(self, data, labels):
        tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy")
        tree = tree.fit(data, labels)
        return tree_classifier(tree)


class perceptron_classifier(abstract_classifier):
    def __init__(self, perceptron):
        self.perceptron = perceptron

    def classify(self, features):
        return self.perceptron.predict(features.reshape(1, -1))[0]


class perceptron_factory(abstract_classifier_factory):
    name = "Perceptron"

    def __init__(self):
        pass

    def train(self, data, labels):
        perceptron = sklearn.linear_model.Perceptron()
        perceptron = perceptron.fit(data, labels)
        return perceptron_classifier(perceptron)


accuracy = {}
error = {}
for i, factory in enumerate([tree_factory, perceptron_factory]):
    accuracy[i], error[i] = evaluate(factory(), num_folds)
    print("%s:" % factory.name, "Accuracy=%f" % accuracy[i], "Error=%f" % error[i])
    print()
import csv
with open("experiments12.csv", 'w') as fd:
    csv_writer = csv.writer(fd)
    for k in accuracy:
        csv_writer.writerow([k+1, accuracy[k], error[k]])

