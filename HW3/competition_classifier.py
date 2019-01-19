from statistics import mean, pvariance

from sklearn import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

from HW3.hw3_utils import *
import math
from HW3.classifier import *

def normalize_data():
    train_features, train_labels, test_features = load_data()
    train_features_normalized = preprocessing.normalize(train_features)
    return train_features_normalized, train_labels


def create_normalized_folds():
    num_folds = 5
    normalized_features, labels = normalize_data()
    split_crosscheck_groups((normalized_features, labels), num_folds)

def evaluate_folds(classifier_factory, folds):
    accuracy, error = evaluate(classifier_factory, folds)
    return accuracy, error


# run main for part 3:
##
##FOLD_FILENAME = "ecg_fold_%d_normalized.data"
##
## normalizing data:
##print("normalizing data")
##normalize_data()
##print("creating normalized folds with 5 folds")
##create_normalized_folds()


## reducing features:

##train_features, train_labels, test_features = load_data()

##print("evaluating folds with ID3")
##split_crosscheck_groups((train_features, train_labels), 5)
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##
##train_features_KBest = SelectKBest(chi2, k=5).fit_transform(abs(train_features), train_labels)
##split_crosscheck_groups((train_features_KBest, train_labels), 5)
##
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##
##
##train_features_KBest = SelectKBest(chi2, k=20).fit_transform(abs(train_features), train_labels)
##split_crosscheck_groups((train_features_KBest, train_labels), 5)
##
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##
##
##train_features_KBest = SelectKBest(chi2, k=80).fit_transform(abs(train_features), train_labels)
##split_crosscheck_groups((train_features_KBest, train_labels), 5)
##
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##
##train_features_KBest = SelectKBest(chi2, k=100).fit_transform(abs(train_features), train_labels)
##split_crosscheck_groups((train_features_KBest, train_labels), 5)
##
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##
##
##train_features_KBest = SelectKBest(chi2, k=120).fit_transform(abs(train_features), train_labels)
##split_crosscheck_groups((train_features_KBest, train_labels), 5)
##
##accuracy, error = evaluate(tree_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)


## using minimum number of samples in leaf = 30
##accuracy, error = evaluate(tree_min_samples_leaf_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)

## using knn with k=1 and ensemble of classifiers:
accuracy, error = evaluate(ensemble_factory(), 5)
print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
