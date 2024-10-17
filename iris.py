import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('IRIS.csv')

# Display dataset shape and information
print(df.shape)
print(df.info())
print(df.head())

# Describe the dataset
df.describe(include='all')

# Plot species frequency
df['species'].value_counts(normalize=True).plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Flower Species')
plt.title('Frequency of Flowers')
plt.show()

# Correlation matrix
corr = df.drop(columns='species').corr()
print(corr)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df, hue='species', diag_kind="hist", corner=True, palette='hls')
plt.show()

# Prepare the data for modeling
target = 'species'
X = df.drop(columns=target)
y = df[target]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Calculate baseline score
y_baselinescore = df['species'].value_counts(normalize=True).iloc[0]  # Use iloc for positional access
print("Baseline score:", y_baselinescore)

# Train models
log = LogisticRegression(max_iter=200)
forest = RandomForestClassifier(random_state=42)

log.fit(X_train, y_train)
forest.fit(X_train, y_train)

# Feature importance
features = X_train.columns
importances = forest.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail().plot(kind='barh')
plt.xlabel('Importance Ratio')
plt.ylabel('Attributes')
plt.title('Feature Importances')
plt.show()

# Evaluate models (Optional)
y_pred_log = log.predict(X_test)
y_pred_forest = forest.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))
