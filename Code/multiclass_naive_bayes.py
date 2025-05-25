import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import GridSearchCV 
import warnings
# import time

# start_time = time.time()


warnings.filterwarnings('ignore')

class NaiveBayes:

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

        self.test_ids = self.df_test['id']
        self.df_test = self.df_test.drop('id',axis =1)


    def train_test_split_nb(self):
        self.X = self.df_train.drop(['id','Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults'], axis= 1)
        self.y = self.df_train[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults']]
        
        self.X = self.X.select_dtypes(include=['number'])
        
        # Scaling features of X and testing dataset
        scaler = MinMaxScaler() 
        self.X= scaler.fit_transform(self.X)
        self.df_test = scaler.fit_transform(self.df_test)

        # Splitting training dataset into Training and Validaton set (70/30 ratio)
        X_train, X_val, y_train, self.y_val = train_test_split(self.X, self.y, train_size= 0.7, random_state= 42)

        # Parameters which has multiple values 
        param_grid = {
        'estimator__var_smoothing': [1e-6, 1e-5, 1e-3, 1e-2, 1e+1]
        }

        # Hyperparameter: var_smoothing for GaussianNaiveBayes multiclass Dataset using One vs Rest classifier
        model =OneVsRestClassifier(GaussianNB())
        self.grid = GridSearchCV(model, param_grid, scoring='f1_micro', cv=5)
        self.grid.fit(X_train, y_train)

        print("Best smoothing:", self.grid.best_params_) # Best Parameter
        print("Best smoothing:", self.grid.best_score_) # Best Score

        self.y_pred_val = self.grid.predict(X_val)
        self.y_pred_proba = self.grid.predict_proba(X_val)

        # Testing dataset Predicting and Probabilities 
        y_pred_test = self.grid.predict(self.df_test)
        y_test_proba= self.grid.predict_proba(self.df_test)



    def plots(self):
        auc_scores = []
        plt.figure(figsize=(10, 6))
        for defect_type in self.labels:
            # MLPClassifier's predict_proba returns probabilities
            val_preds = self.y_pred_proba[:, 1] # Get probability of the positive class
            auc = roc_auc_score(self.y_val[defect_type], val_preds)
            plt.bar(defect_type, auc, color='skyblue')
            plt.text(defect_type, auc + 0.02, f"{auc:.2f}", ha='center', va='bottom', fontsize=10)
            print(f"Validation AUC for {defect_type}: {auc:.4f}")
            auc_scores.append(auc)
        print(f"\nAverage Validation AUC: {np.mean(auc_scores):.4f}")
        plt.ylim(0, 1.0)
        plt.ylabel('AUC Score')
        plt.title('AUC Scores for Each Steel Plate Fault Class for Naive Bayes Multiclass')
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Hyperparameter Tuning for Var smoothing
        smoothing_values = self.grid.cv_results_['param_estimator__var_smoothing'].data
        mean_scores = self.grid.cv_results_['mean_test_score']
        sorted_indices = np.argsort(smoothing_values)
        smoothing_values = smoothing_values[sorted_indices]
        mean_scores = mean_scores[sorted_indices]
        y_min = mean_scores.min() - 0.01
        y_max = mean_scores.max() + 0.01
        plt.figure(figsize=(6, 4))
        plt.plot(smoothing_values, mean_scores, marker='o', linestyle='-', color='blue')
        for i, score in enumerate(mean_scores):
            plt.text(smoothing_values[i], score + 0.001, f"{score:.3f}", ha='center', fontsize=8)
        plt.xscale('log')
        plt.ylim(y_min, y_max)
        plt.xlabel('var_smoothing ')
        plt.ylabel('Mean CV Score')
        plt.title('var_smoothing using mean CV score for Naive Bayes Multiclass')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        acc_score = accuracy_score(self.y_val, self.y_pred_val)

        return acc_score

# nb = NaiveBayes()
# nb.statistics()
# nb.train_test_split_nb()
# nb.metrics()
# nb.plots()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
