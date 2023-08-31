import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import load_img
from keras.utils import img_to_array



train_data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_data.flow_from_directory("train_data", target_size=(224, 224), batch_size=32, subset='training')
val_generator = train_data.flow_from_directory("val_data", target_size=(224, 224), batch_size=32, subset='validation')

# Defining the CNN architecture
model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=val_generator)

test_data = ImageDataGenerator(rescale=1./255)
test_generator = test_data.flow_from_directory("test_data", target_size=(224, 224), batch_size=32)

test_loss, test_accuracy = model.evaluate(test_generator)

# Saving the model
model.save('sunflower_cnn_model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


img_path = "image1.jpg"
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Using the model to predict the class of the image
prediction = model.predict(img_array)
if prediction[0][0] > prediction[0][1]:
    print('The image contains a healthy sunflower leaf.')
else:
    print('The image contains a leaf with Alternaria leaf blight.')