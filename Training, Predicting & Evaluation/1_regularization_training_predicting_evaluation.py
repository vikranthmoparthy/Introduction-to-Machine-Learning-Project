"""
This file implements the training, regularization (through TSCV) and testing all in one. A lot of the code that we used
was taken from week 3's practical: Logistic Regression and week 6's practical: Regularization.
However, we also had to implement new functionality, including time-series cross validation (TSCV, explained in report),
cross-validation averaging and solver reproducibility. For this, we had to read parts of Scikit-Learn and Pandas documentation. 
The sources:
    Time-Series Cross Validation: https://shorturl.at/x7B7v
    Cross Validation Averaging: https://shorturl.at/fYNwO
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss

SEED = 42 #Define global seed to remove any randomness in the solver

#We load data and perform a strict chronological split
#We DO NOT use a train/test split (which shuffles randomly) because this project involves Time Series Data
def load_and_split_data(filepath):
    df = pd.read_csv(filepath)
    
    #Here, we define our predictors and target variable
    feature_cols = ['avg_load', 'max_solar', 'avg_wind_onshore', 'avg_temp', 'gas_price', 'price_yesterday']
    X = df[feature_cols]
    y = df['target']

    #Here, we start implementing the chronological splitting
    #Last 20% of the year = Final Test Set (Locked Vault).
    #First 80% of the year = Training/Validation Set (Used for learning & tuning).
    test_size = int(len(df) * 0.20)
    split_index = len(df) - test_size

    #We use integer-based indexing to slice the data
    X_train_full = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train_full = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    return X_train_full, X_test, y_train_full, y_test, X.columns

#Here, we perform hyperparameter tuning using TSCV. 
#Instead of one validation set, we use 5 rolling windows to test each C value (explained further in report)
def find_best_c_cv(X, y, C_values):
    tscv = TimeSeriesSplit(n_splits=5) #Intializing time-series split (5 splits)
    mean_val_accuracies = []
    
    #Print table of c values, with corresponding accuracy and log loss
    print(f"{'C Value':<10} | {'Avg Val Accuracy':<20} | {'Avg Val Log Loss':<20}") 
    print("-" * 55)

    #Similiar to practical 6, we iterate through every c value (regularization strength) The c-values we use are defined in main().
    for C in C_values:
        fold_accuracies = []
        fold_losses = []
        
        #Initiliaze model with current C
        lr = LogisticRegression(
            C=C, 
            solver='lbfgs', 
            class_weight="balanced", 
            max_iter=5000,
            random_state=SEED
        )

        #Here, we implement the cross-validation loop
        for train_index, val_index in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            lr.fit(X_train_fold, y_train_fold)
            
            fold_accuracies.append(lr.score(X_val_fold, y_val_fold))
            fold_losses.append(log_loss(y_val_fold, lr.predict_proba(X_val_fold)))

        avg_acc = np.mean(fold_accuracies)
        avg_loss = np.mean(fold_losses)
        mean_val_accuracies.append(avg_acc)

        print(f"{C:<10} | {avg_acc:<20.4f} | {avg_loss:<20.4f}")

    best_index = np.argmax(mean_val_accuracies)
    best_C = C_values[best_index]
    
    print(f"\nBest C based on TimeSeries CV: {best_C}")
    return best_C

def train_and_evaluate_final_model(X_train, y_train, X_test, y_test, best_C):
    final_model = LogisticRegression(
        C=best_C, 
        solver='lbfgs', 
        class_weight="balanced", 
        max_iter=5000,
        random_state=SEED
    )
    final_model.fit(X_train, y_train)
    
    predictions = final_model.predict(X_test)
    
    print("\n Final Test Set Evaluation")
    print(classification_report(y_test, predictions))
    
    return final_model

def format_feature_importance(model, feature_names):
    coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
    
    coefficients['Abs_Influence'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Abs_Influence', ascending=False)
    
    print("\nFeature Coefficients:")
    print(coefficients[['Feature', 'Coefficient']].to_string(index=False))

def main():
    X_train_full, X_test, y_train_full, y_test, feature_names = load_and_split_data('processed_training_data.csv')

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_C = find_best_c_cv(X_train_full, y_train_full, C_values)

    final_model = train_and_evaluate_final_model(X_train_full, y_train_full, X_test, y_test, best_C)

    format_feature_importance(final_model, feature_names)

if __name__ == "__main__":
    main()