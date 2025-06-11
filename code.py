

We’ll use the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by European cardholders in September 2013.

📦 1. Install Required Packages                                                                                                                            pip install pandas scikit-learn matplotlib seaborn                                                                                                    🧠 2. Code: Credit Card Fraud Detection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('creditcard.csv')  # Make sure the dataset is in your working directory

# Inspect data
print(df.head())
print(df['Class'].value_counts())

# Separate features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (stratify to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Optional: visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.show()
📊 Notes
Highly imbalanced dataset: Fraud cases are extremely rare. Use stratified sampling and metrics like recall, precision, F1-score rather than accuracy.

Scaling is important: Because most features are PCA components (V1 to V28), scaling helps.

You can replace RandomForestClassifier with more sophisticated models like XGBoost, LightGBM, or even deep learning models.

🔐 Advanced Improvements (Optional)
Use SMOTE to balance classes.

Try Anomaly Detection (e.g., Isolation Forest, One-Class SVM).

Apply cross-validation.

Optimize hyperparameters with GridSearchCV or Optuna.