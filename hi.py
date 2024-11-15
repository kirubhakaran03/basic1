from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset
df = pd.read_csv(r'C:\Users\water\Desktop\resume\weather_forecast_data.csv')
print(df.head())

# Separate features and target
X = df.drop('Rain', axis=1)  # Replace 'target_column' with your target column name
y = df['Rain']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)



# Predict and evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(y_test)

#with open('logistic_regression_model.pkl', 'wb') as file:
    #pickle.dump(model, file)






