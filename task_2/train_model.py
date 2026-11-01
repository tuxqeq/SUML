from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Step 1
iris_data = load_iris()

print(f"Keys in the iris dataset object: {list(iris_data.keys())}")

X = iris_data.data
y = iris_data.target

columns  = iris_data.feature_names

print('Number of samples: ', X.shape)
print('Number of features: ', len(columns))
print('Feature names: ', columns)
print('Target names: ', iris_data.target_names)

# Step 2
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# Step 3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Step 4
y_pred = neigh.predict(X_test)

print("Actual:   ", y_test[:10])
print("Predicted:", y_pred[:10])

# Step 5
accuracy = neigh.score(X_test, y_test)
print("Accuracy: ", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Step 6
joblib.dump(neigh, "model.joblib")

# Step 7
loaded_model = joblib.load("model.joblib")
y_pred_loaded = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)

print("Original model accuracy:", accuracy)
print("Loaded model accuracy:  ", loaded_accuracy)
