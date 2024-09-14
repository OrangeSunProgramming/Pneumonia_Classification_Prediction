import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Define batch size, image dimensions, and load model
batch_size = 32
img_height = 180
img_width = 180

model = tf.keras.models.load_model(f"/content/drive/MyDrive/Pneumonia_classification/chest_xray_models/chest_xray_prediction_model{model_number}.keras")

# Get the true labels and predictions
y_true = []
y_pred = []

# Iterate through validation dataset batches
for images, labels in val_ds:
    y_true.extend(labels.numpy())  # Ground truth labels
    predictions = model.predict(images)
    
    # If using sigmoid output for binary classification, apply threshold
    predicted_labels = (predictions > 0.5).astype("int32").flatten()  # Adjust threshold if needed
    
    # If using softmax, use argmax for binary classification
    # predicted_labels = np.argmax(predictions, axis=1)  # Uncomment if using softmax

    y_pred.extend(predicted_labels)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate precision, recall, and F1-score
precision = precision_score(y_true, y_pred, average='binary')  # binary classification
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# You can still evaluate accuracy using the model's evaluate function
accuracy = model.evaluate(val_ds)