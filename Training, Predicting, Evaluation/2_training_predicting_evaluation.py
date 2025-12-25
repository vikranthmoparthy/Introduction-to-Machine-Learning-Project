import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('processed_training_data.csv')

X = df[['avg_load', 'max_solar', 'avg_wind_onshore', 'avg_temp', 'gas_price', 'price_yesterday']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

logmodel = LogisticRegression(class_weight="balanced")
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logmodel.coef_[0]
})

coefficients['Abs_Influence'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Abs_Influence', ascending=False)

print("\nModel Intercept:", logmodel.intercept_[0])
print("\nFeature Coefficients:")
print(coefficients[['Feature', 'Coefficient']].to_string(index=False))