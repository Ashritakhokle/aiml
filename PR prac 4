# DIAMOND PRICE PREDICTION USING DECISION TREE CLASSIFIER

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv("/content/archive (10).zip")
print("First 5 rows:\n", data.head())
print("\nColumns:\n", data.columns)

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Target column (Price)
target_col = 'Price(in US dollars)'

# Create price categories (Low, Medium, High)
data['Price_Category'] = pd.qcut(data[target_col], q=3, labels=['Low', 'Medium', 'High'])

# Split features and target
X = data.drop(columns=[target_col, 'Price_Category'])
y = data['Price_Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# Evaluation
print("\nDecision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=['Low', 'Medium', 'High']).plot(cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Feature Importance
plt.figure(figsize=(10,6))
plt.barh(X.columns, dt_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Decision Tree - Feature Importance (Diamond Price Categories)")
plt.show()
