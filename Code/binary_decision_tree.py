import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# import time
# start_time = time.time()

class DecisionTree:

    def __init__(self):
        # Setting Training and Testing path
        train_path = '../Dataset/binary_train.csv'
        test_path = '../Dataset/binary_test.csv'
        
        self.df_train = pd.read_csv(train_path)
        self.df_test = pd.read_csv(test_path)
        
    def statistics_dataset(self):
        # Compute whether dataset has any null value and its datatypes
        # self.df_train.info()
        self.df_train.columns
        self.df_train.isna().sum()
        self.df_train.isnull().sum()
        # self.df_test.info()
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

    def train_test_split(self):
        self.y = self.df_train_new['Response']
        self.X = self.df_train_new.drop('Response', axis= 1)
        self.X = self.X.select_dtypes(include=['number'])
        
        # Scaling features of X and testing dataset
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)
        self.df_test_new = scaler.fit_transform(self.df_test)

        n_splits = 5
        training_accuracies  = []
        validation_accuracies = []

        # Splitting training dataset into Training and Validaton set (70/15/15 ratio)
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, train_size=0.7, random_state= 42, stratify= self.y) 
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 15% val, 15% test

        # Kfold cross validation where splits are 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"Using KFold with {n_splits} splits.\n")
        # Hyperparameter: criterion and max_depth for DecisionTreeClassifier
        self.model = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, class_weight= 'balanced')

        for fold_num, (train_index, val_index) in  enumerate (kf.split(self.X_val)):
            
            print(f"--- Fold {fold_num + 1}/{n_splits} ---")
           
            X_fold_train, X_fold_val = self.X_val[train_index], self.X_val[val_index]
            # y_fold_train, y_fold_val = y_val[train_index], y_val[val_index]
            y_fold_train, self.y_fold_val = self.y_val.iloc[train_index], self.y_val.iloc[val_index]
           
            # print(f"  Training data for this fold: X={X_fold_train.shape}, y={y_fold_train.shape}")
            # print(f"  Validation data for this fold: X={X_fold_val.shape}, y={self.y_fold_val.shape}")
            
            self.model.fit(X_fold_train, y_fold_train)
            y_train_pred = self.model.predict(X_fold_train)
            self.y_val_pred = self.model.predict(X_fold_val)

            acc_y_val = accuracy_score(self.y_fold_val, self.y_val_pred)
            validation_accuracies.append(acc_y_val) # Validation accuracy
            acc_y_train = accuracy_score(y_fold_train, y_train_pred)
            training_accuracies.append(acc_y_train)

            print(f"Average accuracy train{fold_num + 1}:  {np.mean(acc_y_train):.2f}")
            print(f"Average accuracy val {fold_num + 1}:  {np.mean(acc_y_val):.2f}")

        self.mean_train_accuracy = np.mean(acc_y_train)
        self.std_train_accuracy = np.std(acc_y_train)
        self.mean_val_accuracy = np.mean(acc_y_val)
        self.std_val_accuracy = np.std(acc_y_val)

        self.y_pred_test = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, self.y_pred_test) # Testing accuracy
        # print(f"Testing Accuracy: {test_accuracy}")

        self.proba_test= self.model.predict_proba(self.X_test)
        # print(f"Test Probabilities:\n {self.proba_test} ")

        # df_proba = pd.DataFrame(proba, columns=["Class 0", "Class 1"])  # Modify based on number of classes
        self.class_0 = sum(self.proba_test[self.proba_test < 0.5])
        self.class_1 = sum(self.proba_test[self.proba_test >= 0.5])

        # Testing dataset Prediction
        pred_test = self.model.predict(self.df_test_new)
        
        plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, marker='o', linestyle='-', label="Validation Accuracy")
        plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, marker='o', linestyle='-', label="Training Accuracy")
        plt.axhline(y=np.mean(validation_accuracies), color='r', linestyle='--', label="Mean Validation Accuracy")
        plt.axhline(y=np.mean(training_accuracies), color='g', linestyle='--', label="Mean Training Accuracy")
        plt.xlabel("Fold Number")
        plt.ylabel("Accuracy")
        plt.title("K-Fold Cross-Validation Accuracy on Validation Set Decision Tree Binary Class")
        plt.legend()
        plt.grid(True)
        plt.show()


    def plots(self):        
        bars = plt.bar(["Class 0", "Class 1"], [self.class_0, self.class_1], color=["blue", "red"])
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Class 0 and Class 1 in Test Data Decision Tree Binary Class")
        plt.legend(bars, ["Class 0 (values < 0.5)", "Class 1 (values ≥ 0.5)"]) 
        plt.show()


        # Calculate the Area Under the ROC Curve (AUC)
        test_probability = self.proba_test[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, test_probability)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve Decision Tree Binary Class')
        plt.legend(loc="lower right")
        plt.grid(True) 
        plt.show()
        print(f"AUC Score: {roc_auc:.2f}")

        return roc_auc

    def metrics(self):
        print(f"Classification Report for Testing Dataset:\n {classification_report(self.y_test, self.y_pred_test)}")
        print(f"Classification Report for Validation Dataset:\n {classification_report(self.y_fold_val, self.y_val_pred)}")
        print(f"Mean Squared error for Testing Dataset: {mean_squared_error(self.y_test, self.y_pred_test)}\n")
        print(f"Confusion matrix for Testing Dataset:\n {confusion_matrix(self.y_test, self.y_pred_test)}")



# dt = DecisionTree()

# dt.one_hot_encoding()
# dt.train_test_split()
# dt.plots()
# dt.metrics()

# end_time = time.time()
# print(f"Execution Time: {end_time - start_time:.2f} seconds")
