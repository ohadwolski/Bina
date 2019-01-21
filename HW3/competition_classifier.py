from statistics import mean, pvariance
from scipy.interpolate import interp1d
from sklearn import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from hw3_utils import *
import math
from classifier import *
import matplotlib.pyplot as plt

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


# run tests for part 3:
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
##accuracy, error = evaluate(ensemble_factory(), 5)
##print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)


## using adaBoost with 100 estimators of stumps:
#accuracy, error = evaluate(adaBoostFactory(), 5)
#print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)


## using reduced data with knn:
#accuracy, error = evaluate(knn_reduced_samples_factory(1), 5)
#print("%s:" % tree_factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)



#train_features, train_labels, test_features = load_data()
#train_labels = np.array(train_labels)
#train_features = np.array(train_features)

#split_crosscheck_groups((train_features, train_labels), 5)

## Test for viewing features as continuous graph:
##plt.figure()
##plt.plot((train_features[train_labels == True]).T)
##plt.show()
##plt.figure()
##plt.plot((train_features[train_labels == False]).T)
##plt.show()

## Removing samples that are "dead", that do not go over a certain threshold
## Scaling all other samples so they fit in 187 samples


##train_features_scaled, train_labels_scaled = scale_data(train_features, train_labels)
##plt.figure()
##plt.plot((train_features_scaled[train_labels_scaled == True]).T)
##plt.show()
##plt.figure()
##plt.plot((train_features_scaled[train_labels_scaled == False]).T)
##plt.show()




## checking knn with preprocessing of data, for different k':


##factory = knn_with_preprocessing_factory(1)
##accuracy, error = evaluate(factory, 5)
##print("%s:" % factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##factory = knn_with_preprocessing_factory(3)
##accuracy, error = evaluate(factory, 5)
##print("%s:" % factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##factory = knn_with_preprocessing_factory(5)
##accuracy, error = evaluate(factory, 5)
##print("%s:" % factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##factory = knn_with_preprocessing_factory(7)
##accuracy, error = evaluate(factory, 5)
##print("%s:" % factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)
##factory = knn_with_preprocessing_factory(13)
##accuracy, error = evaluate(factory, 5)
##print("%s:" % factory.name, "Accuracy=%f" % accuracy, "Error=%f" % error)











