from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. View basic info
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Target names:", target_names)

# 3. Put into DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# 4. Visualization - scatter plot of petal length vs. petal width
plt.figure(figsize=(8,6))
for species in range(3):
    subset = df[df['species'] == species]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], label=target_names[species])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width (by species)')
plt.legend()
plt.show()

# 5. Visualization - histogram for petal length
df['petal length (cm)'].hist(by=df['species'], figsize=(12,4), bins=15)
plt.suptitle('Distribution of Petal Length by Species')
plt.show()

# 6. Split dataset for model training (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train a simple classifier (Logistic Regression)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# 8. Make predictions
y_pred = clf.predict(X_test)

# 9. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred, target_names=target_names))