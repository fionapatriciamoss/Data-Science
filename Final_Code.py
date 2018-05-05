# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:04:58 2018

@author: fmoss1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from scipy.stats import itemfreq
from sklearn import svm, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, roc_curve, auc, hamming_loss, mean_squared_error
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import interp
from itertools import cycle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import seaborn as sns
sns.set(style="whitegrid")

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import itertools

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

dataset_week1 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 4/ML Project/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv')
#dataset_week2 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 4/ML Project/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week2.csv')
#dataset_week3 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 4/ML Project/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week3.csv')
#dataset_week4 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 4/ML Project/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv')

df1 = pd.DataFrame()
def preprocessing(dataset):
    #Removing irrelevant columns
    new = dataset.drop(['Date first seen', 'Flows', 'Tos', 'attackID', 'attackDescription'], axis = 1)

    #converting MB to bytes
    new['Bytes'] = np.where(new['Bytes'].str[-1].str.contains('M') == True, (new['Bytes'].str[:-1].astype(float))*1000000, new['Bytes'])
    
    #splitting the Source Port and Destination Port into 3 categories
    new['Src Pt'] = pd.cut(new['Src Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
    new['Dst Pt'] = pd.cut(new['Dst Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
    
    #finding the subnet of Source and Destination IP Addresses
    new['Src IP Addr'] = new['Src IP Addr'].str.split('_').str[0]
    new['Dst IP Addr'] = new['Dst IP Addr'].str.split('_').str[0]
    
    #Classifying the class label to attack and non-attack
    new = new[new['class'] != 'victim']
    new['class'] = np.where(new['class'].str.contains('attacker') == True, 1, 0)
    
    #counting the frequencies of each IP subnet
#    counts_src = new['Src IP Addr'].value_counts()
#    counts_dst = new['Dst IP Addr'].value_counts()
#    
    #storing rows with IP Address frequency > 1000
#    df = new[new['Src IP Addr'].isin(counts_src[counts_src > 1000].index)]
#    df = new[new['Dst IP Addr'].isin(counts_dst[counts_dst > 1000].index)]
#    
#    #removed rows (<= 1000)
#    df1 = new[new['Src IP Addr'].isin(counts_src[counts_src <= 1000].index)]
#    df1 = df1.append(new[new['Dst IP Addr'].isin(counts_src[counts_src <= 1000].index)])
    
    #group IP Addresses based on frequencies
#    df['Src IP Addr'] = df.groupby('Src IP Addr')['Src IP Addr'].transform('count')
#    df['Dst IP Addr'] = df.groupby('Dst IP Addr')['Dst IP Addr'].transform('count')

    #add another label column with combined class and attackType labels 
#    new['attack'] = new["class"].map(str) + new["attackType"]
    
    #return the resultant cleaned dataset
    return new

dataset_week1 = preprocessing(dataset_week1)
#dataset_week2 = preprocessing(dataset_week2)
#dataset_week3 = preprocessing(dataset_week3)
#dataset_week4 = preprocessing(dataset_week4)

#dataframes = [dataset_week1, dataset_week2, dataset_week3, dataset_week4]

result = dataset_week1

result = result.values

data_attack = result[np.where(result[:,-2]>0.5),:][0]
data_normal = result[np.where(result[:,-2]<0.5),:][0]

attack_bf = data_attack[data_attack[:,-1] == 'bruteForce']
attack_dos = data_attack[data_attack[:,-1] == 'dos']
attack_ping = data_attack[data_attack[:,-1] == 'pingScan']
attack_port = data_attack[data_attack[:,-1] == 'portScan']

#data_attack_down_sampled = np.r_[attack_bf[0:15,:], attack_dos[0:8389,:], attack_ping[0:32,:], attack_port[0:1564,:]]
#data_down_sampled = np.r_[data_attack_down_sampled, data_normal[0:94062,:]]

data_attack_down_sampled = np.r_[attack_bf[0:2,:], attack_dos[0:838,:], attack_ping[0:4,:], attack_port[0:156,:]]
data_down_sampled = np.r_[data_attack_down_sampled, data_normal[0:9406,:]]

#data_attack_down_sampled = np.r_[attack_bf[0:8,:], attack_dos[0:4195,:], attack_ping[0:16,:], attack_port[0:782,:]]
#data_down_sampled = np.r_[data_attack_down_sampled, data_normal[0:47031,:]]

result = data_down_sampled

result_features_copy = result[:, 0:9]
result_features = result_features_copy
result_labels = result[:, 9]

# =============================================================================
# ENCODING CATEGORICAL FEATURES TO NUMERIC FEATURES
# =============================================================================

label_En = LabelEncoder()
result[:, 1] = label_En.fit_transform(result[:, 1])
result[:, 2] = label_En.fit_transform(result[:, 2])
result[:, 3] = label_En.fit_transform(result[:, 3])
result[:, 4] = label_En.fit_transform(result[:, 4])
result[:, 5] = label_En.fit_transform(result[:, 5])
result[:, 8] = label_En.fit_transform(result[:, 8])
result[:, 9] = label_En.fit_transform(result[:, 9])
result[:, 10] = label_En.fit_transform(result[:, 10])

#result_labels = result[:, 9].astype('int32')

one_hot_encoder = OneHotEncoder(categorical_features = [1, 3, 5, 8])
result_features = one_hot_encoder.fit_transform(result_features).toarray()
#result_features = np.column_stack((result_features, result[:, 10]))

#one_hot_encoder = OneHotEncoder(categorical_features = 'all')
#result_labels = one_hot_encoder.fit_transform(result_labels).toarray()
result_labels = result[:, 9:11]

np.savetxt('down_sampled_data.csv', np.hstack((result_features, result_labels)), delimiter=",", fmt = '%s')

# =============================================================================
# run from here
# =============================================================================

result = np.genfromtxt('C:/Users/Fiona/Downloads/Semester 4/Data Science in Security/down_sampled_data.csv', delimiter = ',')

result_features = result[:, :-2]
result_labels = result[:, -2:]

# =============================================================================
# feature selection
# =============================================================================

def etc(sample_train, label_train, sample_test, label_test):
    
    # build a forest to determine feature importances
    forest = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=6, n_estimators=100, random_state = 0)    
    forest.fit(sample_train, label_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(sample_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(sample_train.shape[1]), importances[indices],
           color="b", align="center")
    plt.xticks(range(sample_train.shape[1]), indices)
    plt.xlim([-1, sample_train.shape[1]])
    plt.show()
    
    sfm = SelectFromModel(forest, threshold = 0.000005)
    sfm.fit(sample_train, label_train)
    
#    for f in sfm.get_support(indices=True):
#        print (names[f])
        
    X_important_train = sfm.transform(sample_train)
    X_important_test = sfm.transform(sample_test)
    
    # Create a new random forest classifier for the most important features
    clf_important = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=6, n_estimators=100)    
    # Train the new classifier on the new dataset containing the most important features
    clf_important.fit(X_important_train, label_train)
    
    # Apply The Full Featured Classifier To The Test Data
    y_pred = forest.predict(sample_test)
    
    # View The Accuracy Of Our Full Feature (4 Features) Model
    print("Before feature selection accuracy:", accuracy_score(label_test, y_pred))
    print("Before feature selection F1 Score:", f1_score(label_test, y_pred))
    print("Before feature selection Confusion Matrix:", confusion_matrix(label_test, y_pred))
    
    # Apply The Full Featured Classifier To The Test Data
    y_important_pred = clf_important.predict(X_important_test)
    
    # View The Accuracy Of Our Limited Feature (2 Features) Model
    print("After feature selection accuracy:", accuracy_score(label_test, y_important_pred))
    print("After feature selection F1 Score:", f1_score(label_test, y_important_pred))
    print("After feature selection Confusion Matrix:", confusion_matrix(label_test, y_important_pred))
    
#    plot_confusion_matrix(metrics('Extra Trees', label_test, y_important_pred, num_classes)[1], classes=names)
    return X_important_train, X_important_test

# =============================================================================
# METRICS AND THEIR PLOTS
# =============================================================================

# Metrics
def metrics(strategy, label_test, label_pred, num_classes):
    a_score = accuracy_score(label_test,label_pred)
    a = confusion_matrix(label_test, label_pred)
    p = np.zeros(num_classes)

    r = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    falsealarms = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tp = np.zeros(num_classes)
    for i in range(0, num_classes):
        p[i] = a[i][i]/(a[0][i] + a[1][i] + a[2][i] + a[3][i] + a[4][i])
        r[i] = a[i][i]/(a[i][0] + a[i][1] + a[i][2] + a[i][3] + a[i][4])
        f1[i] = (2 * p[i] * r[i])/ (p[i] + r[i]) 
        falsealarms[i] = (a[0][i] + a[1][i] + a[2][i] + a[3][i] + a[4][i]) - a[i][i]
        fn[i] = (a[i][0] + a[i][1] + a[i][2] + a[i][3] + a[i][4]) - a[i][i]
        tp[i] = a[i][i]
    print (strategy ,'metrics: \naccuracy:', a_score, '\nconfusion matrix: \n', a, '\nprecision: ', p, '\nrecall: ', r, '\nfalse positives(false alarms): ', falsealarms, '\nfalse negatives: ', fn, '\ntrue positives: ', tp)
    metrics = [a_score, a, p, r, f1, falsealarms, fn, tp]
    return metrics

# =============================================================================
# plot confidence matrix
# =============================================================================
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# =============================================================================
# WEIGHTED LEAST SQUARE
# =============================================================================
weight_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]

#weight_arr = [1/np.std(result)*10000, 1/np.var(result), 0.09631511800978098]

#weight_arr = [i for i in range(0.9)]

loss = []*5

epsilon = 0.1
   
sample = result[:,0:-2]
label = result[:,-2:]

#sample = sm.add_constant(sample)

col1 = np.ones((len(sample), 1))
sample = np.hstack((col1, sample))

X, X_test, Y, Y_test = train_test_split(sample, label,
                                                stratify=result[:,-1], 
                                                test_size=0.5)

label_test_AT=Y_test[:, 1]
Y=Y[:, 0]
Y_test=Y_test[:, 0]

sample_train_fs, sample_test_fs = etc(X, Y,  X_test, Y_test)

X = sample_train_fs
X_test = sample_test_fs


X = X.astype(float)
Y = Y.astype(float)

x_transpose = X.transpose()
y_transpose = Y.transpose()[np.newaxis]

x_transpose = x_transpose.astype(float)
y_transpose = y_transpose.astype(float)

#model = sm.OLS(Y,X)
#results = model.fit()



lm = LinearRegression()
lm.fit(X, Y)
linear_pred_test = lm.predict(X_test)
threshold = 0
linear_pred_test = (linear_pred_test > threshold).astype(int)

linear_pred_train = lm.predict(X)
threshold = 0
linear_pred_train = (linear_pred_train > threshold).astype(int)
#linear_pred[linear_pred == 0] = -1
mse = accuracy_score(Y_test, linear_pred_test)
print ("MSE Linear", mse)
print (accuracy_score(Y_test, linear_pred_test))
print (confusion_matrix(Y_test, linear_pred_test))

plt.figure()
plot_confusion_matrix(confusion_matrix(Y_test, linear_pred_test), classes=['Normal', 'Attack'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#residual_test = Y_test - linear_pred_test
#residual_train = Y - linear_pred_train
#
#residual = np.concatenate([residual_train, residual_test])

# Plot outputs
#plt.scatter(X_test[:,0], Y_test,  color='black')
#plt.plot(X_test[:,0], linear_pred, color='blue', linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()

# =============================================================================
# plot residuals
# =============================================================================
#plt.scatter(lm.predict(X), lm.predict(X) - Y, c = 'b', s = 40, alpha = 0.5)
#plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c = 'g', s = 40)
#plt.hlines(y = 0, xmin = 0, xmax = 50)

# =============================================================================
# weight part
# =============================================================================
for w in range(0, len(weight_arr)):
    var2 = result[:, -2]
    var1 = np.where(var2 >= 0.5, 1, weight_arr[w])   
    
    split_length = int (len(var2)/2)

    weight = np.diag(var1)
    weight_train = np.diag(var1[0:split_length])
    weight_test = np.diag(var1[split_length:])
    
# =============================================================================
#     Weighted Least Square
# =============================================================================
    
    iden = np.identity(len(x_transpose))
    
    beta = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(x_transpose, weight_train), X) + epsilon*iden), x_transpose), weight_train), Y)
    
    X_test = X_test.astype(float)
    Y_test = Y_test.astype(float)
    
    temp1 = np.subtract(np.matmul(X_test, beta) , Y_test)
    temp_transpose = temp1.transpose()
    
    loss = (np.dot(np.dot(temp_transpose, weight_test), temp1))/len(Y_test)
    
    mult = np.dot(weight_test, X_test)
    
    print ("Total MSE:", float(loss))

    label_pred = np.dot(X, beta)
    
    
    label_pred = ((label_pred - min(label_pred)) * (1))/(max(label_pred) - min(label_pred)) + 0
    threshold = (max(label_pred) + min(label_pred))/2
    label_pred = (label_pred > threshold).astype(int)
#    label_pred = 1 - label_pred
    
    print (accuracy_score(Y_test, label_pred))
    print (confusion_matrix(Y_test, label_pred))

    plt.figure()
    plot_confusion_matrix(confusion_matrix(Y_test, label_pred), classes=['Normal', 'Attack'], normalize=True,
                      title='Normalized confusion matrix')
    
    plt.show()


#    num_classes = 2
#
#    # ROC curve
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(num_classes):
#        fpr[i], tpr[i], _ = roc_curve(label_test[:, i], label_score[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    
#    # Compute micro-average ROC curve and ROC area
#    fpr["micro"], tpr["micro"], _ = roc_curve(label_test.ravel(), label_score.ravel())
#    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#    
#    # Compute macro-average ROC curve and ROC area
#    
#    # First aggregate all false positive rates
#    lw = 2
#    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
#    
#    # Then interpolate all ROC curves at this points
#    mean_tpr = np.zeros_like(all_fpr)
#    for i in range(num_classes):
#        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#    
#    # Finally average it and compute AUC
#    mean_tpr /= num_classes
#    
#    fpr["macro"] = all_fpr
#    tpr["macro"] = mean_tpr
#    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#    
#
#    ### Plot all ROC curves
#    plt.figure()
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]),
#             color='deeppink', linestyle=':', linewidth=4)
#    
#    plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["macro"]),
#             color='navy', linestyle=':', linewidth=4)
#    
#    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#    for i, color in zip(range(num_classes), colors):
#        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                 label='ROC curve of class {0} (area = {1:0.2f})'
#                 ''.format(i, roc_auc[i]))
#    
#    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic for multi-class')
#    plt.legend(loc="lower right")
#    plt.show()
#
#
#def plot_confusion_matrix(cm, classes,
#                      normalize=False,
#                      title='Confusion matrix',
#                      cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#    
#    print(cm)
#    
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#    
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#plt.tight_layout()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
# =============================================================================
#                                 LEVEL 2
#                        PREDICTION OF ATTACK TYPE
# =============================================================================

#consider only ypred = 1 (attack) indexes
idx_attack = np.where(label_pred==1)[0]

#find those indexes and store them as Label for L2
result_labels_L2=label_test_AT[idx_attack]

# Find the indexed calue from sample_test_fs as features for L2
result_features_L2=sample_test_fs[idx_attack]

# split data into training and test sample
sample_train_L2, sample_test_L2, label_train_L2, label_test_L2 = train_test_split(result_features_L2, result_labels_L2, test_size=0.5, random_state=0)

##nn
#nn = MLPClassifier()
#nn.fit(sample_train_L2, label_train_L2)
#nn_pred = nn.predict(sample_test_L2)
##threshold = 0
##nn_pred = (nn_pred > threshold).astype(int)
##nn_pred[nn_pred == 0] = -1
#accuracy = accuracy_score(label_test_L2, nn_pred) 
#f1_val = f1_score(label_test_L2, nn_pred, average = None)
#con_matrix = confusion_matrix(label_test_L2, nn_pred)
#print ("accuracy of neural networks:", accuracy)
#print ("F1 score:", f1_val)
#print ("Confusion Matrix:", con_matrix)
#print ("Precision:", con_matrix[1][1] / (con_matrix[0][1]+con_matrix[1][1]))
#print ("Recall:", con_matrix[1][1] / (con_matrix[1][0]+con_matrix[1][1]))
#print ("False positives:" ,con_matrix[0][1] / (con_matrix[0][1]+con_matrix[1][1]))

##rf
rf = RandomForestClassifier()
rf.fit(sample_train_L2, label_train_L2)
rf_pred = rf.predict(sample_test_L2)
#threshold = 0
#rf_pred = (rf_pred > threshold).astype(int)
#rf_pred[rf_pred == 0] = -1
accuracy = accuracy_score(label_test_L2, rf_pred)
f1_val = f1_score(label_test_L2, rf_pred, average = None)
con_matrix = confusion_matrix(label_test_L2, rf_pred) 
print ("accuracy of random forest:", accuracy)
print ("F1 score:", f1_val)
print ("Confusion Matrix:", con_matrix)
print ("Precision:", con_matrix[1][1] / (con_matrix[0][1]+con_matrix[1][1]))
print ("Recall:", con_matrix[1][1] / (con_matrix[1][0]+con_matrix[1][1]))
print ("False positives:" ,con_matrix[0][1] / (con_matrix[0][1]+con_matrix[1][1]))


##svm
svm_classifier1 = svm.SVC(C=1)
svm_classifier1.fit(sample_train_L2, label_train_L2)
svm_pred1 = svm_classifier1.predict(sample_test_L2)
#threshold = 0.5
#svm_pred1 = (svm_pred1 > threshold).astype(int)
#svm_pred1[svm_pred1 == 0] = 0
accuracy = accuracy_score(label_test_L2, svm_pred1)
f1_val = f1_score(label_test_L2, svm_pred1, average = None)
con_matrix = confusion_matrix(label_test_L2, svm_pred1)
print ("Accuracy score of svm:", accuracy)
print ("F1 score:", f1_val)
print ("Confusion Matrix:", con_matrix)
print ("Precision:", con_matrix[1][1] / (con_matrix[0][1]+con_matrix[1][1]))
print ("Recall:", con_matrix[1][1] / (con_matrix[1][0]+con_matrix[1][1]))
print ("False positives:" ,con_matrix[0][1] / (con_matrix[0][1]+con_matrix[1][1]))


##logistic reg
LR = LogisticRegression()
LR.fit(sample_train_L2, label_train_L2)
ls_pred1 = LR.predict(sample_test_L2)
#phi_thresh = 0
#ls_pred1 = (ls_pred1 > phi_thresh).astype(int)
#ls_pred1[ls_pred1 == 0] = -1
accu_score_lr = accuracy_score(label_test_L2, ls_pred1)
f1_score_LR_lr = f1_score(label_test_L2, ls_pred1, average = None)
prec_sc_lr = precision_score(label_test_L2, ls_pred1, average = None)
con_matrix = confusion_matrix(label_test_L2, svm_pred1)
print ("Accuracy score of lr:", accu_score_lr)
print ("F1 score:", f1_score_LR_lr)
print ("Confusion Matrix:", con_matrix)
print ("Precision:", con_matrix[1][1] / (con_matrix[0][1]+con_matrix[1][1]))
print ("Recall:", con_matrix[1][1] / (con_matrix[1][0]+con_matrix[1][1]))
print ("False positives:" ,con_matrix[0][1] / (con_matrix[0][1]+con_matrix[1][1]))




