import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import matplotlib.pyplot as plt
import json


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tfds.disable_progress_bar()
from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy().squeeze()
    return image

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255  # Tensor: rescaling images to be between 0 and 1
    image = image.numpy().squeeze()

    image = np.expand_dims(image, axis=0)

    ps = model.predict(image)[0]  # ps is a list of lists, we have only one, we lelect that one

    probabilities = np.sort(ps)[-top_k:len(ps)]
    prbabilities = probabilities.tolist()
    m = prbabilities.index(max(prbabilities))
    classes = np.argpartition(ps, -top_k)[-top_k:]
    classes = classes.tolist()
    names = [class_names.get(str(i + 1)).capitalize() for i in (classes)]

    ps_cl = list(zip(prbabilities, names))
    print(ps_cl)
    print(names[m])
    return probabilities, names

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised = True, with_info = True)

training_set, validation_set, testing_set = dataset['train'], dataset['validation'], dataset['test']

num_training_examples = dataset_info.splits['train'].num_examples
num_validation_examples = dataset_info.splits['validation'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

num_classes = dataset_info.features['label'].num_classes

shape_images = dataset_info.features['image'].shape

for image, label in training_set.take(3):
    image = image.numpy().squeeze()
    label = label.numpy()


with open('label_map.json', 'r') as f:
    class_names = json.load(f)


# create pipeline
batch_size = 32
image_size = 224


#image_gen_train = ImageDataGenerator(rescale = 1./255,
                                     #rotation_range = 45,
                                     #width_shift_range = 0.2,
                                     #height_shift_range=0.2,
                                     #shear_range=0.2,
                                     #zoom_range=0.2,
                                     #horizontal_flip=True,
                                     #fill_mode='nearest')





training_batches = training_set.cache().shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = testing_set.map(format_image).batch(batch_size).prefetch(1)


# building and training
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size, 3))
feature_extractor.trainable = False
layer_neurons = [650, 330, 250]
dropout_rate = 0.2
model = tf.keras.Sequential()
model.add(feature_extractor)
model.add(tf.keras.layers.Dense(102, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

with tf.device('/GPU:0'):
    EPOCHS = 20
    history = model.fit(training_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)

#plotting the graph
#for training and validation comparision

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
#plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')

#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, training_loss, label='Training Loss')
#plt.plot(epochs_range, validation_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()

loss, accuracy = model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

saved_keras_model_filepath = './safi.h5'
model.save(saved_keras_model_filepath)

loaded_model = tf.keras.models.load_model("safi.h5", custom_objects={'KerasLayer':hub.KerasLayer})
loaded_model.summary()

image_path = 'fl1.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

image_size = 224

image_path = 'fl1.jpg'
predict(image_path, loaded_model, 5)




