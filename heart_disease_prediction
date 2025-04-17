import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#creating dataframe
df = pd.read_csv('heart.csv')

# get main information
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("correlation matrix")
plt.show()

X = df.drop('target', axis=1)  
y = df['target'] # 0-no disease, 1-disease

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# training / test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#model
model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


#model validation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#confusion matrix visualization with seaborn
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show()