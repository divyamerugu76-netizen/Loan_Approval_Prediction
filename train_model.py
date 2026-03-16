import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
# data = pd.read_csv("dataset.csv")

data = pd.read_csv("dataset.csv")
data.columns = data.columns.str.strip()
# Features and target
X = data[['Salary','Credit Score','Loan Amount','Employment Status']]
y = data['Approved']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model Trained and Saved Successfully")