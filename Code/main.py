import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
from binary_decision_tree import DecisionTree
from binary_svm import SupportVectorMachine
from binary_knn import KNN
from binary_perceptron import Perceptron1
from multiclass_neural_netwrok import NeuralNetwork
from multiclass_decision_tree import DecisionTreeMulticlass
from multiclass_naive_bayes import NaiveBayes
from multiclass_svm import SupportVectorMachineMulticlass
import warnings
import time

warnings.filterwarnings('ignore')

print("Health Insurance Cross Validation Binary Dataset Training and Testing")

# Decision Tree Binary Dataset
start_time_bdt = time.time()
print("------------------------------------ \n")
print("Binary Class Decision Tree ALgorithm Training and Testing \n \n")
binarydt = DecisionTree()

binarydt.one_hot_encoding()
binarydt.train_test_split()
roc_auc_bdt = binarydt.plots()
binarydt.metrics()
print("------------------------------------ \n")
end_time_bdt = time.time()
print(f"Execution time for Decision Tree (Binary): {end_time_bdt - start_time_bdt}")

# KNN Binary Dataset
start_time_knn = time.time()
print("------------------------------------ \n")
print("Binary Class KNN ALgorithm Training and Testing \n \n")
knn = KNN()

knn.one_hot_encoding()
knn.knn_training_split()
roc_auc_knn =knn.plots()
knn.metrics()

print("------------------------------------ \n")
end_time_knn = time.time()
print(f"Execution time for KNN: {end_time_knn - start_time_knn}")

# Perceptron Binary set
start_time_perceptron = time.time()
print("------------------------------------ \n")
print("Binary Class Perceptron ALgorithm Training and Testing \n \n")
perceptron = Perceptron1()

perceptron.one_hot_encoding()
perceptron.train_test_split_perceptron()
perceptron.metrics()
roc_auc_perceptron =perceptron.plots()

print("------------------------------------ \n")
end_time_perceptron = time.time()
print(f"Execution time for Perceptron: {end_time_perceptron - start_time_perceptron}")

# SVM Binary_set
start_time_svm = time.time()
print("------------------------------------ \n")
print("Binary Class SVM ALgorithm Training and Testing \n \n")

svm = SupportVectorMachine()
svm.one_hot_encoding()
svm.train_test_split_svm()
svm.metrics()
roc_auc_svmb =svm.plots()

print("------------------------------------ \n")
end_time_svm = time.time()
print(f"Execution time for SVM (Binary): {end_time_svm - start_time_svm}")

# Comparing Performace of each model:
model_names = ['DecisionTree', 'KNN', 'Perceptron', 'SVM']
roc_auc_scores = [roc_auc_bdt, roc_auc_knn, roc_auc_perceptron, roc_auc_knn]


plt.figure(figsize=(6, 4))
bars = plt.bar(model_names, roc_auc_scores  , color=['blue', 'pink', 'skyblue', 'purple'])
plt.ylabel('ROC_AUC')
plt.title('Model Performance Comparison for binary class')

# Add accuracy labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.ylim(0, 1)  # Because accuracy ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



print("Steel Plate Defect Prediction Multiclass Dataset Training and Testing")
# Multiclass DecisionTree
start_time_dtm = time.time()
print("------------------------------------ \n")
print("Multiclass Decision Tree ALgorithm Training and Testing \n \n")

dtm = DecisionTreeMulticlass()
dtm.statistics()
accuracy_dtm = dtm.train_test_split_dt()
dtm.metrics()
dtm.plots()

print("------------------------------------ \n")
end_time_dtm = time.time()
print(f"Execution time for Decision Tree (Multiclass): {end_time_dtm - start_time_dtm}")

# Multiclass SVM
start_time_svm1 = time.time()
print("------------------------------------ \n")
print("Multiclass SVM ALgorithm Training and Testing \n \n")

svm1 = SupportVectorMachineMulticlass()
svm1.statistics()
svm1.train_test_split_svm()
accuracy_svm = svm1.metrics()
svm1.plots()

print("------------------------------------ \n")
end_time_svm1 = time.time()
print(f"Execution time for SVM (Multiclass): {end_time_svm1 - start_time_svm1}")

# Multiclass Neural Network
start_time_nn = time.time()
print("------------------------------------ \n")
print("Multiclass Neural Network ALgorithm Training and Testing \n \n")

nn = NeuralNetwork()
nn.statistics()
nn.train_test_split_nn()
nn.plots()
accuracy_NN = nn.metrics()

print("------------------------------------ \n")
end_time_nn = time.time()
print(f"Execution time for Neural Network (Multiclass): {end_time_nn - start_time_nn}")

# Multiclass Naive Bayes
start_time_nb = time.time()
print("------------------------------------ \n")
print("Multiclass Naive Bayes ALgorithm Training and Testing \n \n")

nb = NaiveBayes()
nb.statistics()
nb.train_test_split_nb()
accuracy_NB = nb.metrics()
nb.plots()

print("------------------------------------ \n")
end_time_nb = time.time()
print(f"Execution time for Naive Bayes (Multiclass): {end_time_nb - start_time_nb}")


# Comparing Performace of each model:
model_names = ['DecisionTree', 'SVM', 'Neural Network', 'Naive Bayes']
roc_auc_scores = [accuracy_dtm, accuracy_svm, accuracy_NN, accuracy_NB]


plt.figure(figsize=(6, 4))
bars = plt.bar(model_names, roc_auc_scores  , color=['Red', 'blue', 'pink', 'skyblue'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison for multiclass')

# Add accuracy labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.ylim(0, 1)  # Because accuracy ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()