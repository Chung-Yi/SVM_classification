import numpy as np
import pandas as pd
import os
import joblib
import glob
import cv2
import random
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import shuffle
from xgboost import XGBClassifier


IMG_PATH = 'car/'
IMG_SIZE = 100
IMGS = glob.glob(IMG_PATH + '**/*.jpg')
random.shuffle(IMGS)
Categories = ['front', 'profile']

X = []
Y = []

def create_data():
    for img in IMGS:
        category = img.split('\\')[1]
        label = Categories.index(category)
        image = cv2.imread(img)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        X.append(image)
        Y.append(label)


def xgb_(x_train, x_test, y_train, y_test):

    x_train = x_train.reshape(len(x_train), 100*100*3)
    x_test = x_test.reshape(len(x_test), 100*100*3)
    # y_train = y_train.reshape(len(y_train), 100*100*1)
    # y_test = y_test.reshape(len(y_test), 100*100*1)

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.02, 0.05]
    }

    folds = 3

    param_comb = 20

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic', silent=True, nthread=2, tree_method='gpu_hist', eval_metric='auc')
    # xgb = XGBClassifier()
    # cv_results = cross_val_score(xgb, x_train, y_train,
    #                cv = 2, scoring='accuracy', n_jobs = -1, verbose = 1)
    # print(cv_results)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', cv=skf.split(x_train, y_train), verbose=3, random_state=1001)

    print('Start training XGBboost...')
    # xgb.fit(x_train, y_train, verbose=True)
    # y_pred = xgb.predict(x_test)

    random_search.fit(x_train, y_train)

    y_pred = random_search.predict(x_test)

    acc = metrics.accuracy_score(y_pred, y_test)

    print(f'accuracy: {acc}')

def svm_(x_train, x_test, y_train, y_test):

    # SVM
    param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly', 'linear']}

    svc = SVC(probability=True)

    grid_search = GridSearchCV(svc, param_grid, n_jobs=-1)

    print('Start to search the best parameters...')

    grid_result = grid_search.fit(x_train, y_train)

    print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print('Start to train...')

    svc_bestparam = SVC(probability=False, C=grid_result.best_params_['C'], gamma=grid_result.best_params_['gamma'], kernel=grid_result.best_params_['kernel'])

    svc_bestparam.fit(x_train, y_train)

    # y_pred = svc_bestparam.predict_proba(x_test)
    y_pred = svc_bestparam.predict(x_test)


    acc =metrics.accuracy_score(y_pred, y_test)

    print(f'accuracy: {acc}')

    print(y_pred)

    filename = 'model/svm_classifier.sav'

    joblib.dump(svc_bestparam, filename)



def main():
    create_data()
    x = np.array(X).reshape(len(X),-1)
    y = np.array(Y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    #######################################
    svm_(x_train, x_test, y_train, y_test)
    # xgb_(x_train, x_test, y_train, y_test)



if __name__ == '__main__':
    main()