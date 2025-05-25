import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# import time

# start_time = time.time()


warnings.filterwarnings('ignore')

class SupportVectorMachineMulticlass:

    def __init__(self):
        # Setting Training and Testing path
        train_path = '../Dataset/multiclass_train.csv'
        test_path = '../Dataset/multiclass_test.csv'
        
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        
    def dataset_info(self):
        # Compute whether dataset has any null value and its datatypes
        self.df_train.info()
        self.df_train.columns
        self.df_train.isna().sum()
        self.df_train.isnull().sum()
        self.df_test.info()
        self.df_test.columns
        self.df_test.isnull().sum()

    def statistics(self):
        correlation = self.df_train.corr() # Checking correlation with all other fatures
        
        # Y labels
        self.labels = self.df_train[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults']]

        self.y_counts = pd.DataFrame()

        for column in self.labels.columns:
            count = self.labels[column].value_counts()
            self.y_counts[column] = count

        print(self.y_counts) # Counts how many value each label has 

        self.df_test = self.df_test.drop('id',axis =1)


    def train_test_split_svm(self):
        self.X = self.df_train.drop(['id','Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults'], axis= 1)
        self.y = self.df_train[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults']]
        
        self.X = self.X.select_dtypes(include=['number'])
        
        # Scaling features of X and testing dataset
        scaler = StandardScaler()
        self.X= scaler.fit_transform(self.X)
        self.df_test = scaler.fit_transform(self.df_test)

        # Splitting training dataset into Training and Validaton set (70/30 ratio)
        X_train, X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, train_size= 0.7, random_state= 42)
        # Hyperparameters: C and kernel for support vector machine multiclass dataset
        # param ={'C': [0.0001, 0.001, 0.01, 0.1, 1 ,10]}
        support_vector = SVC(C= 1, kernel= 'rbf', max_iter= 1000, probability= True, class_weight='balanced')
        sv = OneVsRestClassifier(support_vector)
        # sv = GridSearchCV(model, param, cv= 5)
        sv.fit(X_train, self.y_train)
        self.y_pred_train = sv.predict(X_train)
        self.y_pred_val = sv.predict(X_val)
        self.y_val_probability = sv.predict_proba(X_val) 
        # print(f"Best Parameter: {sv.best_params_}")
        # Testing dataset Predicting and Probabilities 
        pred_test = sv.predict(self.df_test)
        proba_test= sv.predict_proba(self.df_test)

    def plots(self):
        auc_scores = []
        plt.figure(figsize=(10, 6))
        for defect_type in self.labels:
            # MLPClassifier's predict_proba returns probabilities 
            val_preds = self.y_val_probability[:, 1] # Get probability of the positive class
            auc = roc_auc_score(self.y_val[defect_type], val_preds)
            plt.bar(defect_type, auc, color='skyblue')
            plt.text(defect_type, auc + 0.02, f"{auc:.2f}", ha='center', va='bottom', fontsize=10)
            print(f"Validation AUC for {defect_type}: {auc:.4f}")
            auc_scores.append(auc)
        print(f"\nAverage Validation AUC: {np.mean(auc_scores):.4f}")
        plt.ylim(0, 1.0)
        plt.ylabel('AUC Score')
        plt.title('AUC Scores for Each Steel Plate Fault Class SVM multiclass ')
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        avg_val_score = np.mean(auc_scores)

        # Hyperparameter tuning for C values
        C_values = np.logspace(-5, 1, 7)
        accuracies = []
        acc_train = []

        for c in C_values:
            acc = accuracy_score(self.y_val, self.y_pred_val)
            acc1 = accuracy_score(self.y_train, self.y_pred_train)
            accuracies.append(acc)
            acc_train.append(acc1)
            print(f"C={c:.1e} -> Accuracy: {acc:.4f}")


        plt.figure(figsize=(8, 5))
        plt.semilogx(C_values, accuracies, marker='o', label = 'Validation Accuracy')
        plt.semilogx(C_values, acc_train, marker='o', label = 'Training Accuracy')
        plt.xlabel('C')
        plt.ylabel('Validation Accuracy')
        plt.title('Manual Hyperparameter Tuning of C for SVM multiclass')
        plt.grid(True)
        plt.legend()
        plt.show()

        return avg_val_score

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        print(f"Classification Report for Validation Dataset:\n {accuracy_score(self.y_val, self.y_pred_val)}")
        acc_socre = accuracy_score(self.y_val, self.y_pred_val)

        return acc_socre
        
# svm = SupportVectorMachineMulticlass()
# svm.statistics()
# svm.train_test_split_svm()
# svm.metrics()
# svm.plots()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
