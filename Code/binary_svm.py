import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from os.path import join, dirname, abspath, exists
import warnings
# from sklearn.model_selection import GridSearchCV
# import time
# start_time = time.time()

warnings.filterwarnings('ignore')

class SupportVectorMachine:

    def __init__(self):
        # Setting Training and Testing path
        train_path = '../Dataset/binary_train.csv'
        test_path = '../Dataset/binary_test.csv'
        
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        
    def statistics_dataset(self):
        # Compute whether dataset has any null value and its datatypes
        self.df_train.info()
        self.df_train.columns
        self.df_train.isna().sum()
        self.df_train.isnull().sum()
        self.df_test.info()
        self.df_test.columns
        self.df_test.isnull().sum()

    def unique_values(self, df, columns):
        # Check unique values for multiple Categorical Features
        df = df.copy()
        return {column: list(df[column].unique()) for column in columns}
    
    def binary_encoding(self, df, column, positive_label):
        # Binary Encoding for binary value features
        df[column] = df[column].apply(lambda x: 1 if x == positive_label else 0 )
        return df
    
    def ordinal_encoding(self, df, column, ordering):
        # Ordinal Encoding for ordinal value features
        df[column] = df[column].apply(lambda x: ordering.index(x))
        return df
    
    def one_hot_encoding(self):
        # one hot encoding for binary and ordinal features
        Categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        self.unique_values(self.df_train, Categorical_features)
        vehicle_age_ordering  = ['< 1 Year', '1-2 Year', '> 2 Years']

        self.df_train = self.binary_encoding(self.df_train, 'Gender', 'Male')
        self.df_test = self.binary_encoding(self.df_test, 'Gender', 'Male')
        self.df_train = self.binary_encoding(self.df_train, 'Vehicle_Damage', 'Yes')
        self.df_test = self.binary_encoding(self.df_test, 'Vehicle_Damage', 'Yes') 
    
        self.df_train = self.ordinal_encoding(self.df_train, 'Vehicle_Age', vehicle_age_ordering)
        self.df_test = self.ordinal_encoding(self.df_test, 'Vehicle_Age', vehicle_age_ordering)
        self.df_train_new = self.df_train.drop('id', axis = 1)
        self.df_test = self.df_test.drop('id', axis = 1)

        # print(f"Training Dataset: {self.df_train.info()}")
        # print(f"Testing Dataset: {self.df_test.info()}")
    
    def train_test_split_svm(self):
        self.y = self.df_train_new['Response']
        self.X = self.df_train_new.drop('Response', axis= 1)
        self.X = self.X.select_dtypes(include=['number'])
        
        # Scaling features of X and testing dataset
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.df_test_new = scaler.fit_transform(self.df_test)

        # Splitting training dataset into Training and Validaton set (70/30 ratio)
        X_train, X_val, y_train, self.y_val = train_test_split(self.X, self.y, test_size= 0.3, random_state= 42, stratify=self.y) 
        # Hyperparameter: C and kernel for SVC binary dataset
        # param = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        support_vector = SVC(C =0.1, kernel= 'rbf', max_iter= 1000, probability= True, class_weight='balanced')
        # support_vector = GridSearchCV(model, param, cv=5)
        support_vector.fit(X_train, y_train)
        # print(f"Best Parameter: {support_vector.best_params_}")

        Training_score = support_vector.score(X_train, y_train)
        # print(f"Training Score for SVM: {Training_score}")
        
        y_train_pred = support_vector.predict(X_train)    
        self.y_pred_val = support_vector.predict(X_val)
        Validation_score = support_vector.score(X_val, self.y_val) # Validation Score
        print(f"Validation Score for SVM: {Validation_score}")

        self.proba_val= support_vector.predict_proba(X_val)
        # print(f"Testing Probablities for SVM: {self.proba_val}")
        self.proba_log_test = support_vector.predict_log_proba(X_val)
        # print(f"Log Testing Probalities for SVM: {self.proba_log_test}")
        
        # df_proba = pd.DataFrame(self.proba_val, columns=["Class 0", "Class 1"])  # Modify based on number of classes
        self.class_0 = sum(self.proba_val[self.proba_val < 0.5])
        self.class_1 = sum(self.proba_val[self.proba_val >= 0.5])

        # Testing dataset Prediction
        pred_test = support_vector.predict(self.df_test)


    def plots(self): 
        # Class Distribution 
        plt.figure(figsize=(6, 4))
        bars = plt.bar(["Class 0", "Class 1"], [self.class_0, self.class_1], color=["blue", "red"])
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Class 0 and Class 1 in Test Data SVM binary class")
        plt.legend(bars, ["Class 0 (values < 0.5)", "Class 1 (values ≥ 0.5)"]) 
        plt.show()

        # Calculate the Area Under the ROC Curve (AUC)  
        test_probability = self.proba_val[:, 1]
        y_pred_proba_inverted = 1 - test_probability
        test_log_prob = self.proba_log_test[:, 1]
        y_pred_proba_inverted1 = 1 - test_log_prob
        fpr, tpr, thresholds = roc_curve(self.y_val, y_pred_proba_inverted)
        fpr2, tpr2, thresholds2 = roc_curve(self.y_val, y_pred_proba_inverted1)
        roc_auc = auc(fpr, tpr)
        roc_auc2 = auc(fpr2, tpr2)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot(fpr2, tpr2, color='green', lw=2,linestyle = 'dotted', label='ROC curve (area = %0.2f)' % roc_auc2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for SVM binary class')
        plt.legend(loc="lower right")
        plt.grid(True) 
        plt.show()
        print(f"AUC Score for Test Probabilities: {roc_auc:.2f}")
        print(f"AUC Score for Log Test Probabilities: {roc_auc2:.2f}")

        return roc_auc

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        print(f"Confusion matrix for Validation Dataset:\n {confusion_matrix(self.y_val, self.y_pred_val)}")
       

# svm = SupportVectorMachine()
# svm.one_hot_encoding()
# svm.train_test_split_svm()
# svm.metrics()
# svm.plots()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
