import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
# import time

# start_time = time.time()

warnings.filterwarnings('ignore')

class DecisionTreeMulticlass:

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


    def train_test_split_dt(self):
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
        X_train, self.X_val, y_train, self.y_val = train_test_split(self.X, self.y, train_size= 0.7, random_state= 42)
        
        # Hyperparameters: mini_samples_split and max_depth for decisionTree multiclass dataset
        self.model = DecisionTreeClassifier(criterion= 'gini', min_samples_split= 2, max_depth= 15, class_weight='balanced')

        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        training_accuracy = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {training_accuracy}")

        self.y_pred_val = self.model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, self.y_pred_val) # Accuracy Score for Validation set
        self.y_pred_proba = self.model.predict_proba(self.X_val)
        print(f"Validation Accuracy for training dataset: {val_accuracy}")

        # Testing dataset Predicting and Probabilities 
        pred_test = self.model.predict(self.df_test)
        proba= self.model.predict_proba(self.df_test)

        return val_accuracy

    def plots(self):
        y_true_bin = label_binarize(self.y_val, classes=[0, 1, 2, 3, 4, 5, 6])
        n_classes = y_true_bin.shape[1]

        # Plot ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_pred_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(f"ROC AUC Values for class[{i}]:{roc_auc[i]}")

        # Plot all ROC curves
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve for DecisionTree')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        # Hyperparameter tuning for max_depth and cross-validated Accuracy
        max_depth_values = range(1, 30)
        mean_scores = []
        for depth in max_depth_values:
            model = DecisionTreeClassifier(criterion= 'gini', min_samples_split= 2, max_depth= depth)
            scores = cross_val_score(model, self.X, self.y, cv=5)  # 5-fold CV
            mean_scores.append(scores.mean())

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(max_depth_values, mean_scores, marker='o')
        plt.xlabel('max_depth')
        plt.ylabel('Cross-Validated Accuracy')
        plt.title('Hyperparameter Tuning: max_depth vs Accuracy for Multiclass DecisionTree')
        plt.grid(True)
        plt.show()

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")

# dt = DecisionTreeMulticlass()
# dt.statistics()
# dt.train_test_split_dt()
# dt.metrics()
# dt.plots()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
