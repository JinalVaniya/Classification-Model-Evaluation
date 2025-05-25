import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV 
# import time

# start_time = time.time()

warnings.filterwarnings('ignore')

class NeuralNetwork:

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


    def train_test_split_nn(self):
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

        self.alpha_values = np.logspace(-5, 1, 7) # Alpha values to check best alpha
        self.val_accuracies = []
        # param = {'hidden_layer_sizes': [(128, 64, 32), (64,32), (64,32,16)]}

        # Hyperparameter: hidden_layer_Sizes and alpha for MLPClassifier (Neural Network)
        for alpha in self.alpha_values:
            self.model = MLPClassifier(hidden_layer_sizes = (128, 64, 32), activation= 'relu',solver= 'adam', 
                    max_iter= 200, random_state= 42,learning_rate=  'adaptive', alpha= alpha, tol= 1e-4, n_iter_no_change= 10 )
            # self.model = GridSearchCV(mlp, param_grid= param, cv = 5, scoring= 'f1_micro')
            self.model.fit(X_train, y_train)
            self.y_pred_val = self.model.predict(self.X_val)
            acc = accuracy_score(self.y_val, self.y_pred_val) # Accuracy score for validation dataset
            self.val_accuracies.append(acc)
            print(f"alpha={alpha:.1e} -> Validation Accuracy: {acc:.4f}")

            avg_val_accuracy = np.mean(acc)
            # best_para = self.model.best_params_
            # print(f"Best Parameter for Hidden Layer: {best_para}")

        # Testing dataset Predicting and Probabilities 
        pred_test = self.model.predict(self.df_test)
        proba= self.model.predict_proba(self.df_test)

        return avg_val_accuracy

    def plots(self):
        # Plotting alpha values hyperparameter
        plt.figure(figsize=(8, 5))
        plt.semilogx(self.alpha_values, self.val_accuracies, marker='o')
        plt.xlabel('Alpha')
        plt.ylabel('Mean Cross-Validated Accuracy')
        plt.title('Hyperparameter Tuning: Alpha for MLPClassifier multiclass')
        plt.grid(True)
        plt.show()


    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        auc_scores = []
        for defect_type in self.labels:
            # MLPClassifier's predict_proba returns probabilities for [class 0, class 1]
            val_preds = self.model.predict_proba(self.X_val)[:, 1] # Get probability of the positive class
            auc = roc_auc_score(self.y_val[defect_type], val_preds)
            print(f"Validation AUC for {defect_type}: {auc:.4f}")
            auc_scores.append(auc)
        print(f"\nAverage Validation AUC: {np.mean(auc_scores):.4f}")
        avg_val_score = np.mean(auc_scores)

        return avg_val_score



# nn = NeuralNetwork()
# nn.statistics()
# nn.train_test_split_nn()
# nn.plots()
# nn.metrics()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
