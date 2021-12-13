import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE

from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

#------ Scaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

#---- Get data_pre_train by using numpy
data_pre_train =np.genfromtxt('./../../csv/trill_final_layer_embeddings_training.csv',delimiter=',')
#data_pre_train =np.genfromtxt('./../../csv/trill_layer_19_embeddings_training.csv',delimiter=',')


data_pre_train = np.delete(data_pre_train, 0 ,0) #delete header
row_pre_train, col_pre_train = np.shape(data_pre_train)

x_pre_train = data_pre_train[:,1:col_pre_train-1]
y_pre_train = data_pre_train[:,col_pre_train-1]

#---- Get data_pre_test by using numpy
data_pre_test =np.genfromtxt('./../../csv/trill_final_layer_embeddings_testing.csv',delimiter=',')
#data_pre_test =np.genfromtxt('./../../csv/trill_layer_19_embeddings_testing.csv',delimiter=',')


data_pre_test = np.delete(data_pre_test, 0 ,0) #delete header
row_pre_test, col_pre_test = np.shape(data_pre_test)

x_pre_test = data_pre_test[:,1:col_pre_test-1]
y_pre_test = data_pre_test[:,col_pre_test-1]

#------ Scaler 
scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = QuantileTransformer(output_distribution="normal")

x_pre_train = scaler.fit_transform(x_pre_train)
x_pre_test = scaler.fit_transform(x_pre_test)


#------- 5-fold cross validation
roc_score_rf       = 0
f1_score_weight_rf = 0
f1_score_macro_rf  = 0
f1_score_acc_rf    = 0
auc                = 0
final_specificity  = 0
final_sensitivity  = 0

###### --------------------------------------------new positive data in for loop
k_neighbors_list = [29,19,9,11,18,26,12,4,10,7,8]
#k_neighbors_list = [19,9,3,11,18,16,12,7,10,17,8]
m_neighbors_list = [3,3,3,3,3,3,3,3,3,3,3]
x_train_new_pre = []
y_train_new_pre = []

#x_pos_in_train = x_pre_train[793:]
#y_pos_in_train = y_pre_train[793:]
x_pos_in_train = np.copy(x_pre_train[793:])
y_pos_in_train = np.copy(y_pre_train[793:])
for i in range(70):
    y_pos_in_train[i] = y_pos_in_train[i] - 1
    
y_neg_fake_in_train = y_pos_in_train

#print(x_pos_in_train.shape)

#print(y_neg_fake_in_train.shape)

for i in range(len(k_neighbors_list)):

    x_train_new, y_train_new = SVMSMOTE(sampling_strategy='minority', k_neighbors = int(k_neighbors_list[i]), m_neighbors = int(m_neighbors_list[i])).fit_resample(x_pos_in_train, y_neg_fake_in_train)
    
    #print(y_train_new_1.shape)
    
    #print(y_train_new_1[172:])
    
    y_train_new[172:] = y_train_new[172:] + 1
    
    #print(y_train_new_1[172:].shape)
    
    #print(x_train_new_1[172:,:].shape)
    
    x_train_new_final = x_train_new[172:,:]
    y_train_new_final = y_train_new[172:]
    
    x_train_new_pre.append(x_train_new_final)
    y_train_new_pre.append(y_train_new_final)
    
    
x_train_new_final = np.concatenate((x_train_new_pre), axis=0)
y_train_new_final = np.concatenate((y_train_new_pre), axis=None)


###### ------------------- Training

fold_num = 5
n_fold = 0
for i in range(fold_num): 
    n_fold = n_fold+1
    
    train_index = np.random.RandomState(seed=n_fold).permutation(row_pre_train)
    test_index = np.random.RandomState(seed=42).permutation(row_pre_test)
    
    x_train, x_test = x_pre_train[train_index], x_pre_test[test_index]
    y_train, y_test = y_pre_train[train_index], y_pre_test[test_index]
    
    # To solve UNBALANCED dataset 
    #x_train, y_train = RandomOverSampler(random_state=1).fit_resample(x_train, y_train)
    #x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    #x_train, y_train = SVMSMOTE(sampling_strategy=0.8).fit_resample(x_train, y_train)
    x_train, y_train = SVMSMOTE(sampling_strategy='minority', k_neighbors = 171, m_neighbors = 3).fit_resample(x_train, y_train)
    #x_train, y_train = SVMSMOTE(sampling_strategy='minority').fit_resample(x_train, y_train)

    #x_train, y_train = ADASYN(sampling_strategy='minority').fit_resample(x_train, y_train)
    
    # Concatenate to new pos data
    x_train = np.concatenate((x_train, x_train_new_final), axis=0)
    y_train = np.concatenate((y_train, y_train_new_final), axis=None)
    
    
    x_train, y_train = SVMSMOTE(sampling_strategy='minority').fit_resample(x_train, y_train)
    
    
    # Scale after SMOTE
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
 
    # print(x_train.shape)
    # print(y_train.shape)
    # print(y_train)
 
    
   
    print("\n ============================================ PROCESS FOLD {}...".format(n_fold))
    train_num = len(x_train)
    test_num = len(x_test)

    print("\n ---------- Classifier: Light_GBM, train {} and test {}".format(len(x_train), len(x_test)))

    #classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(4096, 2), random_state=1, learning_rate_init = 0.001, max_iter = 200)
    classifier = LGBMClassifier(learning_rate = 0.03, objective='binary', metric = 'auc', subsample = 0.68, colsample_bytree = 0.28, n_estimators = 10000, subsample_freq = 1, reg_alpha = 0.3)
    classifier.fit(x_train, y_train)
 
    predicted = classifier.predict(x_train)
    expected = y_train
 
    cm = metrics.confusion_matrix(expected, predicted)
    print("Training Confusion Matrix:\n{}".format(cm))
    print("Training Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))
    
    # Metrics in Train
    tn, fp, fn, tp = cm.ravel()
    
    specificity    = tn/(tn+fp)
    sensitivity    = tp/(tp+fn)
    
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    #auc = metrics.auc(fpr, tpr)
    
    print("\nTraining True Negative: {}".format(tn))
    print("\nTraining False Positive: {}".format(fp))
    print("\nTraining False Negative: {}".format(fn))
    print("\nTraining True Positive: {}".format(tp))
    print("\nTraining Specificity: {}".format(specificity))
    print("\nTraining Sensitivity: {}".format(sensitivity))
    print("\nTraining AUC: {}".format(metrics.auc(fpr, tpr)))
    
    #------------------------------------------------------
    
    predicted = classifier.predict(x_test)
    expected = y_test
    cm = metrics.confusion_matrix(expected, predicted)
    print("\nTesting Confusion Matrix:\n{}".format(cm))
    print("Testing Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))
    
    # Metrics in Test
    tn, fp, fn, tp = cm.ravel()
    
    specificity    = tn/(tn+fp)
    sensitivity    = tp/(tp+fn)
    
    fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
    #auc = metrics.auc(fpr, tpr)
    print("\nTesting True Negative: {}".format(tn))
    print("\nTesting False Positive: {}".format(fp))
    print("\nTesting False Negative: {}".format(fn))
    print("\nTesting True Positive: {}".format(tp))
    print("\nTesting Specificity: {}".format(specificity))
    print("\nTesting Sensitivity: {}".format(sensitivity))
    print("\nTesting AUC: {}".format(metrics.auc(fpr, tpr)))
    
    final_specificity = final_specificity + specificity
    final_sensitivity = final_sensitivity + sensitivity
    
    auc = auc + metrics.auc(fpr, tpr) 
    roc_score_rf   = roc_score_rf + roc_auc_score(expected, predicted)
    f1_score_weight_rf = f1_score_weight_rf + f1_score(expected, predicted, average='weighted')
    f1_score_macro_rf  = f1_score_macro_rf + f1_score(expected, predicted, average='macro')
    f1_score_acc_rf    = f1_score_acc_rf + metrics.accuracy_score(expected, predicted)
    
auc                = auc/fold_num
roc_score_rf       = roc_score_rf/fold_num
f1_score_weight_rf = f1_score_weight_rf/fold_num
f1_score_macro_rf  = f1_score_macro_rf/fold_num
f1_score_acc_rf    = f1_score_acc_rf/fold_num
final_specificity = final_specificity/fold_num
final_sensitivity = final_sensitivity/fold_num


print('\n\n ================================== FINAL REPORT: light_GBM')
print(' +Specificity:{}\n +Sensitivity:{}\n +AUC:{}\n +ACC:{}\n +F1_SCORE_WEIGHTED:{}\n +F1_SCORE_MACRO:{}\n'.format(final_specificity, final_sensitivity, auc, f1_score_acc_rf, f1_score_weight_rf, f1_score_macro_rf))





   
# ###### --------------------------------------------new positive data case 1
# x_pos_in_train = x_pre_train[793:]
# y_pos_in_train = y_pre_train[793:]
# for i in range(30):
#     y_pos_in_train[i] = y_pos_in_train[i] - 1
    
# y_neg_fake_in_train = y_pos_in_train

# #print(x_pos_in_train.shape)

# #print(y_neg_fake_in_train.shape)

# x_train_new_1, y_train_new_1 = SVMSMOTE(sampling_strategy='minority', k_neighbors = 29, m_neighbors = 3).fit_resample(x_pos_in_train, y_neg_fake_in_train)

# #print(y_train_new_1.shape)

# #print(y_train_new_1[172:])

# y_train_new_1[172:] = y_train_new_1[172:] + 1

# #print(y_train_new_1[172:].shape)

# #print(x_train_new_1[172:,:].shape)

# x_train_new_1_final = x_train_new_1[172:,:]
# y_train_new_1_final = y_train_new_1[172:]

# print(x_train_new_1_final.shape)
# print(y_train_new_1_final.shape)







 