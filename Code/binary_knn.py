import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from os import mkdir
from sklearn.model_selection import GridSearchCV 
# import time
# start_time = time.time()

class KNN:

    def __init__(self):
        train_path = '../Dataset/binary_train.csv'
        test_path = '../Dataset/binary_test.csv'
        
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        
    def statistics_dataset(self):
        
        self.df_train.info()
        self.df_train.columns
        self.df_train.isna().sum()
        self.df_train.isnull().sum()
        self.df_test.info()
        self.df_test.columns
        self.df_test.isnull().sum()

    def unique_values(self, df, columns):
        df = df.copy()
        return {column: list(df[column].unique()) for column in columns}
    
    def binary_encoding(self, df, column, positive_label):
        df[column] = df[column].apply(lambda x: 1 if x == positive_label else 0 )
        return df
    
    def ordinal_encoding(self, df, column, ordering):
        df[column] = df[column].apply(lambda x: ordering.index(x))
        return df
    
    def one_hot_encoding(self):
        
        Categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        self.unique_values(self.df_train, Categorical_features)
        vehicle_age_ordering  = ['< 1 Year', '1-2 Year', '> 2 Years']


        self.df_train = self.binary_encoding(self.df_train, 'Gender', 'Male')
        self.df_test = self.binary_encoding(self.df_test, 'Gender', 'Male')
        self.df_train = self.binary_encoding(self.df_train, 'Vehicle_Damage', 'Yes')
        self.df_test = self.binary_encoding(self.df_test, 'Vehicle_Damage', 'Yes') 

    
        self.df_train = self.ordinal_encoding(self.df_train, 'Vehicle_Age', vehicle_age_ordering)
        self.df_test = self.ordinal_encoding(self.df_test, 'Vehicle_Age', vehicle_age_ordering)
        self.test_ids = self.df_test['id'].tolist()
        self.df_train_new = self.df_train.drop('id', axis = 1)
        self.df_test = self.df_test.drop('id', axis = 1)

        # print(f"Training Dataset: {self.df_train.info()}")
        # print(f"Testing Dataset: {self.df_test.info()}")

    def knn_training_split(self):
        self.y = self.df_train_new['Response']
        self.X = self.df_train_new.drop('Response', axis= 1)
    
        self.X = self.X.select_dtypes(include=['number'])
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.df_test_new = scaler.fit_transform(self.df_test)

        X_train, X_val, y_train, self.y_val = train_test_split(self.X, self.y, test_size= 0.3, random_state= 42, stratify= self.y) 
        model = KNeighborsClassifier(n_neighbors=10, metric= 'minkowski')
        param = {'n_neighbors': [7, 10, 12, 15]}
        self.knn = GridSearchCV(model, param_grid=param, scoring='f1_micro')
        self.knn.fit(X_train, y_train)

        self.y_pred_val = self.knn.predict(X_val)
        val_accuracy = accuracy_score(self.y_val, self.y_pred_val)
        best_para = self.knn.best_params_
        best_value= self.knn.best_score_
        print(f"Validaion Accuracy: {val_accuracy}")
        print(f"Best Value: {best_value} \n Best N value: {best_para}")

        self.val_proba= self.knn.predict_proba(X_val)
        # print(f"Test Probabilities:\n {self.test_proba} ")

        df_proba = pd.DataFrame(self.val_proba, columns=["Class 0", "Class 1"])  # Modify based on number of classes
        self.class_0 = sum(self.val_proba[self.val_proba < 0.5])
        self.class_1 = sum(self.val_proba[self.val_proba >= 0.5])

        self.pred_test = self.knn.predict(self.df_test_new)
        
    def plots(self):        
        bars = plt.bar(["Class 0", "Class 1"], [self.class_0, self.class_1], color=["blue", "red"])
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Class 0 and Class 1 in Test Data KNN Binary Class")
        plt.legend(bars, ["Class 0 (values < 0.5)", "Class 1 (values ≥ 0.5)"]) 
        plt.show()

        val_probability = self.val_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_val, val_probability)

        # Calculate the Area Under the ROC Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # --- 3. Plot the ROC Curve ---
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Plot the diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve KNN Binary Class')
        plt.legend(loc="lower right")
        plt.grid(True) # Add a grid for better readability
        plt.show()

        print(f"AUC Score: {roc_auc:.2f}")

        return roc_auc

    def metrics(self):
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_val, self.y_pred_val)}")
        print(f"Accuracy for Validation dataset: {accuracy_score(self.y_val, self.y_pred_val)}\n")
        
# knn = KNN()

# knn.one_hot_encoding()
# knn.knn_training_split()
# knn.plots()
# knn.metrics()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
