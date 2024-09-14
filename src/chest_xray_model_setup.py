from chest_xray_visual_gen import *
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model


#Configuring the dataset for performance

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


#Normalizing the data
#Initializing the sequential model

num_classes = len(class_names) #The number of classes

#Defining inception-like module
def inception_module(input_tensor, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
  conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1,1), padding="same", activation="relu")(input_tensor) #1x1 conv

  conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1,1), padding="same", activation='relu')(input_tensor)
  conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3,3), padding="same", activation="relu")(conv_3x3) #1x1 conv followed by 3x3 conv

  conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1,1), padding="same", activation='relu')(input_tensor)
  conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5,5), padding="same", activation="relu")(conv_5x5) #1x1 conv followed by 5x5 conv

  #3x3 max pooling followed by 1x1 conv
  pool_proj = tf.keras.layers.MaxPooling2D((3,3), strides=(1,1), padding="same")(input_tensor)
  pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1,1), padding="same", activation="relu")(pool_proj)

  #Concatenating all branches
  output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj])

  return output


def build_inception_like_model(input_shape=(img_height, img_width, 3)):
  input_tensor = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Conv2D(64, (7,7), strides=(2,2), padding="same", activation="relu")(input_tensor)
  x = tf.keras.layers.MaxPooling2D((3,3),  strides=(2,2), padding="same")(x)

  x = tf.keras.layers.Conv2D(64, (1,1), padding="same", activation="relu")(x)
  x = tf.keras.layers.Conv2D(192, (3,3), padding="same", activation="relu")(x)
  x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

  #First inception-like module
  x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

  #Second inception-like module
  inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

  #Max Pooling
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding="same")(x)

  #Third inception-like module
  x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)

  #Fourth inception-like module
  x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)

  #Global Average Pooling
  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  #Dense layer for binary classification
  output_tensor = tf.keras.layers.Dense(1, activation="sigmoid")(x)

  model = Model(inputs=input_tensor, outputs=output_tensor)
  return model

inception_like_model = build_inception_like_model(input_shape=(img_height, img_width, 3))


#Compiling the model with Adam optimizer
inception_like_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=['accuracy'])

inception_like_model.summary()

#Saving the inception-like model summary
def save_model_summary(model, filename):
  with open(filename, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

save_model_summary(inception_like_model, f'/content/drive/MyDrive/Pneumonia_classification/chest_xray_model_summary/inception_like_model{model_number}_summary.txt')


#Adding an EarlyStopping callback to prevent overfitting by monitoring validation loss.
early_callback = tf.keras.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=1,
	mode="auto",
	restore_best_weights=True)

epochs=100
model_progress = inception_like_model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_callback])


#Saving the model
inception_like_model.save(f"/content/drive/MyDrive/Pneumonia_classification/chest_xray_models/chest_xray_prediction_model{model_number}.keras")


#Visualizing the training results
#accuracy and validation accuracy

acc = model_progress.history['accuracy']
val_accuracy = model_progress.history['val_accuracy']

#loss and validation loss
loss = model_progress.history['loss']
val_loss = model_progress.history['val_loss']

epochs_trained = len(model_progress.history['loss']) #the number of epochs the model took to train
epochs_range = range(epochs_trained)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc='upper right')
plt.title("Training and Validation Loss")
plt.savefig(f"/content/drive/MyDrive/Pneumonia_classification/chest_xray_visuals/model_training_graphs_model{model_number}.png")
plt.show()