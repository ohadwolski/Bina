import sklearn
from HW3.hw3_utils import *


class ThreeDI_classifier_factory(abstract_classifier_factory):
    def train(self, data, labels):
        '''
        train a classifier
        :param data: a list of lists that represents the features that the classifier will be trained with
        :param labels: a list that represents  the labels that the classifier will be trained with
        :return: abstruct_classifier object
        '''

        return ThreeDI_Classifier(data, labels, self.k)


train_features, train_labels, test_features = load_data()
factory = ThreeDI_classifier_factory()
classifier = factory.train(train_features, train_labels)


class ThreeDI_Classifier(abstract_classifier):
    def classify(self, features):
        '''
        classify a new set of features
        :param features: the list of feature to classify
        :return: a tagging of the given features (1 or 0)
        '''

