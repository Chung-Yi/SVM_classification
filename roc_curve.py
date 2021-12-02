import cv2
import numpy as np
import joblib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, recall_score, precision_recall_curve


model_path = 'model/svm_classifier_prob.sav'
IMGS = glob.glob('test/**/*.jpg')
IMG_SIZE = 100
Categories = ['front', 'profile']

X = []
Y = []

thrs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.75, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98, 1.0]

model = joblib.load(model_path)

def roc(y_true, y_pred):

    precision = []
    recall = []

    tpr = np.array([1.0])
    fpr = np.array([1.0])

    p = len(y_true[y_true==0])
    n = len(y_true) - len(y_true[y_true==0])

    for thr in thrs:

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(y_pred)):
            # print(np.argmax(y_pred[i]))
            idx = np.argmax(y_pred[i])

            if idx == 0 and (y_pred[i][idx] > thr) and (y_true[i] == 0):
                tp += 1

            if idx == 0 and (y_pred[i][idx] > thr) and (y_true[i] == 1):
                fp += 1

            if idx == 1 and (y_pred[i][idx] > thr) and (y_true[i] == 0):
                fn += 1

        precision.append((tp/ (tp + fp)) if (tp + fp) else 1)
        recall.append((tp / (tp + fn)) if (tp + fn) else 1)

        tpr = np.insert(tpr, 0, tp/p)
        fpr = np.insert(fpr, 0, fp/n)

    print(f'precision: {precision}')
    print(f'tpr: {tpr}')
    print(f'recall: {recall}')
    print(f'threshold: {thrs}')

    return fpr, tpr

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

def roc_sklearn(y_true, y_pred):
    prob = y_pred[:, 1]
    fpr, tpr, _ = roc_curve(y_true, prob)
    return fpr, tpr

def create_data():
    for img in IMGS:
        category = img.split('\\')[1]
        label = Categories.index(category)
        image = cv2.imread(img)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        X.append(image)
        Y.append(label)

def main():

    # single image
    # img = cv2.imread('car2_.jpg')
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img = img / 255
    # img = np.expand_dims(img, 0)
    # imgs = img.reshape(1, -1)
    # pred = model.predict_proba(imgs)
    # print(max(pred[0]))

    ################################
    create_data()
    x = np.array(X).reshape(len(X),-1)
    y = np.array(Y)

    y_pred = model.predict_proba(x)
    # fpr, tpr = roc_sklearn(y, y_pred)
    fpr, tpr = roc(y, y_pred)
    plot_roc_curve(fpr, tpr)


if __name__ == '__main__':
    main()