
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 70, num_samples),
        'BMI': np.random.uniform(18.5, 35, num_samples),
        'HeartRate': np.random.randint(60, 100, num_samples),
        'CaloriesBurned': np.random.randint(1500, 3000, num_samples),
        'ActivityLevel': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'HealthRisk': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 0: Low Risk, 1: High Risk
    }
    return pd.DataFrame(data)

# Step 2: Preprocess Data
data = generate_synthetic_data()
data['ActivityLevel'] = data['ActivityLevel'].map({'Low': 0, 'Medium': 1, 'High': 2})  # Encode categorical data

X = data[['Age', 'BMI', 'HeartRate', 'ActivityLevel']]
y = data['HealthRisk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Predictive Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 5: Visualize Data
# Heart Rate Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['HeartRate'], bins=20, kde=True, color='blue')
plt.title('Heart Rate Distribution')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Frequency')
plt.show()

# Health Risk by Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['BMI'], hue=data['HealthRisk'], palette='coolwarm', s=100)
plt.title('Health Risk by Age and BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend(title='Health Risk', labels=['Low Risk', 'High Risk'])
plt.show()

# Step 6: Dummy User Input
user_input = pd.DataFrame({
    'Age': [45],
    'BMI': [28],
    'HeartRate': [75],
    'ActivityLevel': [1]
})
risk_prediction = model.predict(user_input)
risk_level = "High Risk" if risk_prediction[0] == 1 else "Low Risk"
print(f"Predicted Health Risk for User: {risk_level}")
