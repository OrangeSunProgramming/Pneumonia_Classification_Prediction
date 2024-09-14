import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

model_number = 25


data_dir = "/content/drive/MyDrive/Pneumonia_classification/chest_xray_dataset/train"

train_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

class_names = train_ds.class_names #There should be two class names
print(f"Class names: {class_names}")

#Visualizing the dataset

plt.figure(figsize=(20, 20))
for images, labels in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[labels[i]])
		plt.axis("off")
plt.savefig(f"/content/drive/MyDrive/Pneumonia_classification/chest_xray_visuals/label_visual_model{model_number}.png")