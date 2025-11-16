# ================================================
# TASK 1: Classical ML with Scikit-learn (Iris)
# ================================================

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Step 1: Load the Iris dataset and convert to pandas DataFrame
iris = load_iris()
# Combine features and target into a single DataFrame for easier inspection
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Target is already numeric (0, 1, 2)

# Step 2: Check for missing values (none expected, but good practice)
print("Missing values per column:\n", df.isnull().sum())
# No missing values → no handling required

# Step 3: Target is already numeric → no encoding needed

# Step 4: Train/test split: 20% test, random_state=42 for reproducibility
X = df.drop('target', axis=1)  # Features
y = df['target']               # Target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train a DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict on test set and compute metrics (macro average)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Print the three metrics
print(f"\nDecision Tree Performance on Iris Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro):    {recall:.4f}")