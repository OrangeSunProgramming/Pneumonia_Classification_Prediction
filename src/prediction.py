import tensorflow as tf
import numpy as np

test_pic_dir = "/content/drive/MyDrive/Pneumonia_classification/example2.jpg"

img = tf.keras.utils.load_img(test_pic_dir, target_size=(180, 180))

#Loading the saved model
model = tf.keras.models.load_model(f"/content/drive/MyDrive/Pneumonia_classification/chest_xray_models/chest_xray_prediction_model{model_number}.keras")

#Class names
class_names = ["NORMAL", "PNEUMONIA"]

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array) #logits when we use from_logits=True to train the model
#probability = tf.nn.sigmoid(logits)
#prediction = round(probability[0][0].numpy())

#print(f"probability: {logits}")
print(f"prediction: {prediction}")

print(f"The patience shows signs of {class_names[round(prediction[0][0])]}")