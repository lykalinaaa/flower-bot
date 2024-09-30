import argparse

import numpy as np

from PIL import Image
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', default='checkpoint')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='fl1.jpg')
    parser.add_argument('--category_names', dest='category_names', default='label_map.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def load_split_data():
    #load oxford flower dataset using tensorflow_datasets
    dataset, dataset_info = tfds.load('oxford_flowers102', shuffle_files=True, as_supervised = True, with_info = False)
    #split dataset
    training_set, test_set, valid_set = dataset['train'], dataset['test'], dataset['validation']
    num_training_examples = dataset_info.splits['train'].num_examples
    return training_set, test_set, valid_set, training_set, num_training_examples

image_size = 224 #Data is normalized and resized to 224x224 pixels as required by the pre-trained networks
def normalize(image, label):

    image = tf.cast(image, tf.float32) #from  unit8 to float32
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 #rescaling images to be between 0 and 1
    return image, label

batch_size = 32 #I choose a smaller batch so that it can comfortably fit in my computer's memo
def batch_data(training_set, test_set, valid_set, num_training_examples):
    training_batches = training_set.cache().shuffle(num_training_examples//4).map(normalize).batch(batch_size)
    test_batches = test_set.cache().map(normalize).batch(batch_size)
    valid_batches = valid_set.cache().map(normalize).batch(batch_size)
    return training_batches, test_batches, valid_batches

def map_data():
    #mapping integer coded labels to flower names
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    return class_names

# Loading the model

#model = "safi.h5"
def load_model(model):
    loaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_model

def predict(image_path, model, top_k):
    #takes a path and opens an image
    image = Image.open(image_path)
    #creates an array of image
    image = np.asarray(image)
    #turns the immage from unit8 to float32
    image = tf.cast(image, tf.float32)
    #Resizes the image
    image = tf.image.resize(image, (image_size, image_size))
    #Tensor: rescaling images to be between 0 and 1
    image /= 255
    #remove single-dimensional entries from the shape of an array.
    image = image.numpy().squeeze()

    #add an extra dimension back
    image = np.expand_dims(image, axis = 0)


    ps = model.predict(image)[0] #ps is a list of lists, we have only one, we lelect that one
    probabilities = np.sort(ps)[-top_k:len(ps)] # short top probabilities
    prbabilities = probabilities.tolist() #create a list for the probabilities
    classes = np.argpartition(ps, -top_k)[-top_k:] # get names of int classes
    classes = classes.tolist() #create a list of int classes
    names = [map_data().get(str(i + 1)).capitalize() for i in (classes)] # get class names
    return prbabilities, names


def get():
    args = parse_args()
    model = load_model(args.checkpoint)

    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k))

    m = probs.index(max(probs))
    val = classes[m]
    return val

print(get())



