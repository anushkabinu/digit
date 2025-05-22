from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = SVC(kernel='rbf', gamma=0.001)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "svm_digit_model.joblib")
joblib.dump(scaler, "digit_scaler.joblib")
print("[✅] Model and scaler saved successfully.")

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Show predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Pred: {y_pred[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
