import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss

def main():
    df = pd.read_csv('processed_training_data.csv')
    X = df[['avg_load', 'max_solar', 'avg_wind_onshore', 'avg_temp', 'gas_price', 'price_yesterday']]
    y = df['target']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.30, random_state=101)

    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    val_accuracies = []
    
    print(f"{'C Value':<10} | {'Val Accuracy':<15} | {'Val Log Loss':<15}")
    print("-" * 45)

    for C in C_values:
        lr = LogisticRegression(C=C, solver='lbfgs', class_weight="balanced", max_iter=5000)
        lr.fit(X_train, y_train)

        val_acc = lr.score(X_val, y_val)
        
        val_prob = lr.predict_proba(X_val)
        val_loss = log_loss(y_val, val_prob)

        val_accuracies.append(val_acc)

        print(f"{C:<10} | {val_acc:<15.4f} | {val_loss:<15.4f}")

    best_index = np.argmax(val_accuracies)
    best_C = C_values[best_index]
    print(f"\nBest C based on Validation Accuracy: {best_C}")

    final_model = LogisticRegression(C=best_C, solver='lbfgs', class_weight="balanced", max_iter=5000)
    final_model.fit(X_train, y_train)
    
    predictions = final_model.predict(X_test)
    
    print("\n Final Test Set Evaluation")
    print(classification_report(y_test, predictions))

    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': final_model.coef_[0]
    })
    coefficients['Abs_Influence'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Abs_Influence', ascending=False)
    
    print("\nFeature Coefficients:")
    print(coefficients[['Feature', 'Coefficient']].to_string(index=False))

if __name__ == "__main__":
    main()