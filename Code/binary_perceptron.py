import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV 
# import time
# start_time = time.time()

warnings.filterwarnings('ignore')

class Perceptron1:

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
    
    def train_test_split_perceptron(self):
        self.y = self.df_train_new['Response']
        self.X = self.df_train_new.drop('Response', axis= 1)
        self.X = self.X.select_dtypes(include=['number'])
        
        # Scaling features of X and testing dataset
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.df_test_new = scaler.fit_transform(self.df_test)

        # Splitting training dataset into Training and Validaton set (70/30 ratio)
        X_train, X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size= 0.3, random_state= 42, stratify= self.y) 
        # Hyperparameters to check the best value
        param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'penalty':['l1','l2']}

        model = Perceptron(class_weight= 'balanced', max_iter=1000)
        clf = GridSearchCV(model, param, scoring='f1_micro', cv=5)
        clf.fit(X_train, self.y_train)
        print(f"Best Alpha Value and Penalty: {clf.best_params_}") # Best Parameters
        print(f"Best SCore: {clf.best_score_}") # Best score
       
        Training_score = clf.score(X_train, self.y_train)
        self.y_pred_train = clf.predict(X_train)
        # print(f"Training score for Perceptron: {Training_score}")
        self.y_pred_val = clf.predict(X_val)
        val_accuracy = clf.score(X_val, self.y_val) # Validation Score
        print(f"Validation set Accuracy: {val_accuracy}")

        # Testing dataset Prediction
        pred_test = clf.predict(self.df_test)
    

    def plots(self):      
        # Class Distribution  
        y_distribution = self.df_train.iloc[:, -1].values 
        class_counts = pd.Series(y_distribution).value_counts().sort_index()
        plt.figure(figsize=(8, 6))
        class_counts.plot(kind='bar')
        plt.xlabel("Class Label")
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Class Labels (y) Perceptron Binary Class")
        plt.xticks(rotation=0) 
        for index, value in enumerate(class_counts):
            plt.text(index, value, str(value), ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--') 
        plt.tight_layout() # 
        plt.show()

        # Calculate the Area Under the ROC Curve (AUC)                
        fpr, tpr, thresholds = roc_curve(self.y_val, self.y_pred_val)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve Percetron Binary Class')
        plt.legend(loc="lower right")
        plt.grid(True) 
        plt.show()
        print(f"AUC Score: {roc_auc:.2f}")

        # Hyperparameter Alpha tuning
        alpha_values = np.logspace(-5, 1, 7)
        accuracies = []
        acc_train = []
        for alpha in alpha_values:
            acc = accuracy_score(self.y_val, self.y_pred_val)
            acc1 = accuracy_score(self.y_train, self.y_pred_train)
            accuracies.append(acc)
            acc_train.append(acc1)
            print(f"C={alpha:.1e} -> Accuracy: {acc:.4f}")
        plt.figure(figsize=(8, 5))
        plt.semilogx(alpha_values, accuracies, marker='o', label = 'Validation Accuracy')
        plt.semilogx(alpha_values, acc_train, marker = 'o', label = 'Training Accuracy' )
        plt.xlabel('Alpha')
        plt.ylabel('Validation and Training Accuracy')
        plt.title('Manual Hyperparameter Tuning of Alpha Perceptron Binary Class')
        plt.grid(True)
        plt.legend()
        plt.show()

        return roc_auc

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        print(f"Confusion Matrix for Validation dataset: \n {confusion_matrix(self.y_val, self.y_val)}\n")


# perceptron = Perceptron1()

# perceptron.one_hot_encoding()
# perceptron.train_test_split_perceptron()
# perceptron.metrics()
# perceptron.plots()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
