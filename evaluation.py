import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = load_model('model.h5')

fixed_size = (128, 128)

# Load the test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Reshape X_test to match the model's input shape
X_test = X_test.reshape(-1, fixed_size[0], fixed_size[1], 1)

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions and true labels to class indices
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
label_names = ["Leisler's", "Pipistrelle", "Myotis"]
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=label_names)
print("Classification Report:")
print(report)
