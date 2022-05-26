'''
Module to create a supervised machine learning
classifier model from a dataset

Author: Jan Rombouts
Date: 25-05-2022
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

class Classifier:
    '''
    Module to create a supervised machine learning
    classifier model from a dataset
    '''
    def __init__(self, X_train, X_test, y_train, y_test, priority='accuracy'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.priority = priority


    def test_model(self, model):
        '''
        Test model on performance metrics
        Parameters:
            Input: sklearn.fit model
            Output: metrics of model
        '''
        model_pred = model.predict(self.X_test)
        metrics = {'conf_mat': confusion_matrix(self.y_test, model_pred),
        'accuracy':accuracy_score(self.y_test, model_pred),
        'recall':recall_score(self.y_test, model_pred),
        'f1':f1_score(self.y_test, model_pred),
        'precision':precision_score(self.y_test, model_pred)}
        return metrics

    def scoring_metrics(self, model_score):
        '''
        Print scoring metrics
        Parameters:
            Input: scores of model
        '''
        print(f'Confusion matrix: \n {model_score["conf_mat"]}')
        print(f'Accuracy: {model_score["accuracy"]}')
        print(f'Recall: {model_score["recall"]}')
        print(f'F1 Score: {model_score["f1"]}')
        print(f'Precision Score: {model_score["precision"]}')


    def dum_model(self):
        '''
        Create dummy model
        '''
        dummy = DummyClassifier(strategy='most_frequent').fit(self.X_train, self.y_train)
        model_score = self.test_model(dummy)
        print('Dummy model metrics:')
        print(self.scoring_metrics(model_score))
        return dummy


    def log_model(self):
        '''
        Create log model
        '''
        if len(self.y_train) < 1000:
            solver = 'liblinear'
        else:
            solver = 'saga'
        C = range(1,21)
        best_model_score = {self.priority: 0}
        for c in C:
            log_model = LogisticRegression(C=c, solver=solver,random_state=14)
            log_model = log_model.fit(self.X_train, self.y_train)
            model_score = self.test_model(log_model)
            if model_score[self.priority] > best_model_score[self.priority]:
                best_model_score = model_score
                best_model = log_model
                best_c = c
        print('Logistic regression model metrics:\n')
        print(f'The optimal log model C score is: {best_c}')
        print(self.scoring_metrics(best_model_score))
        return best_model


    def svc_model(self):
        '''
        Create Support Vector Machine model
        '''
        C = range(1,21)
        kernels = ['linear','poly','rbf','sigmoid']
        params = [(c,kernel) for c in C for kernel in kernels]
        best_model_score = {self.priority: 0}
        for param in params:
            svc_model = SVC(C=param[0],kernel=param[1],random_state=14)
            svc_model.fit(self.X_train, self.y_train)
            model_score = self.test_model(svc_model)
            if model_score[self.priority] > best_model_score[self.priority]:
                best_model_score = model_score
                best_model = svc_model
                best_param = param
        print('Support Vector Machine model metrics:\n')
        print(f'The optimal SVC model parameters are: C = {best_param[0]}, kernel = {best_param[1]}')
        print(self.scoring_metrics(best_model_score))
        return best_model


    def dtr_model(self):
        '''
        Create Decision Tree model
        '''
        max_depth = range(2,10)
        best_model_score = {self.priority: 0}
        for depth in max_depth:
            dtr_model = DecisionTreeClassifier(max_depth=depth)
            dtr_model.fit(self.X_train, self.y_train)
            model_score = self.test_model(dtr_model)
            if model_score[self.priority] > best_model_score[self.priority]:
                best_model_score = model_score
                best_model = dtr_model
                best_depth = depth
        print('Decision Tree Classifier metrics:\n')
        print(f'The best depth for Decision Tree is {best_depth}')
        print(self.scoring_metrics(best_model_score))
        return best_model

    def gnb_model(self):
        '''
        Create Naive Bayes model
        '''
        gnb = GaussianNB().fit(self.X_train, self.y_train)
        model_score = self.test_model(gnb)
        print('Naive Bayes model metrics:')
        print(self.scoring_metrics(model_score))
        return gnb


    def ROC_curve(self, model):
        '''
        Plots ROC curve of a model
        Parameters:
            Input: sklearn.fit model
        '''
        y_score = model.decision_function(self.X_test)
        fpr, tpr, _ = roc_curve(self.y_test, y_score, pos_label=model.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        prec, recall, _ = precision_recall_curve(self.y_test, y_score, pos_label=model.classes_[1])
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)

if __name__ == "__main__":
    df = pd.read_csv('data/parkinsons.data')
    cols = ['MDVP:Fo(Hz)',
       'MDVP:Fhi(Hz)',
       'MDVP:Flo(Hz)',
       'MDVP:RAP',
       'MDVP:Shimmer',
       'RPDE',
       'DFA',
       'spread1',
       'spread2',
       'D2']
    # Create the X-matrix and the y-vector.
    y = np.array(df['status'])
    X = np.array(df[cols])
    # Split the data in train data and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=14)
    model = Classifier(X_train, X_test, y_train, y_test, priority='recall')
    dtr = model.dtr_model()
    dum = model.dum_model()
    log = model.log_model()
    svc = model.svc_model()
