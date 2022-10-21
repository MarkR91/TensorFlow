# Modified Tensorflow code from Coursera min-project course: "TensorFlow for CNNs"
# Code adjusted to run on my laptop CPU. 
# The number of convolution layers where reduced and number of epochs was also reduced (accuracy of this classifer ended up being around 74.77%)

# Aim is to classify an input image of a Border Collie as a 'dog' with Tensorflow.



# import TensorFlow libraries
import tensorflow as tf

# tf.config.set_visible_devices([], 'GPU')

from keras.preprocessing.image import ImageDataGenerator 


#tf.__version__

# Utiltity code for unzipping/extracting the provided training and validation set folders
import os
import zipfile

# Enter the absolute paths for your training and validation zip folders on your computer
local_zip = r'C:/Users/.../.../.../.../Classification/training_set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(r'C:/Users/.../.../.../.../Classification/training_set')

local_zip = r'C:/Users/.../.../.../.../Classification/validation_set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(r'C:/.../.../.../.../Classification/validation_set')

zip_ref.close()



# The contents of the .zip are extracted to the base directory /dataset, which in turn each contain cats and dogs subdirectories.
# Define the 4 directories

# Enter the absolute paths for your training and validation folders on your computer
train_dogs_dir = os.path.join(r'C:/Users/.../.../.../.../Classification/training_set/training_set/dogs')

train_cats_dir = os.path.join(r'C:/.../.../.../.../.../Classification/training_set/training_set/cats')

validation_dogs_dir = os.path.join(r'C:/Users/.../.../.../.../Classification/validation_set/validation_set/dogs')

validation_cats_dir = os.path.join(r'C:/Users/.../.../.../.../Classification/validation_set/validation_set/cats')


# View the file (image) labels of our dataset

train_dogs_names = os.listdir(train_dogs_dir)
print(train_dogs_names[:10])

train_cats_names = os.listdir(train_cats_dir)
print(train_cats_names[:10])

validation_dogs_hames = os.listdir(validation_dogs_dir)
print(validation_dogs_hames[:10])

validation_cats_names = os.listdir(validation_cats_dir)
print(validation_cats_names[:10])


# View the number of Cats and Dogs images in the dataset

print('total training dogs images:', len(os.listdir(train_dogs_dir)))
print('total training cats images:', len(os.listdir(train_cats_dir)))
print('total validation dogs images:', len(os.listdir(validation_dogs_dir)))
print('total validation cats images:', len(os.listdir(validation_cats_dir)))


# View sample of pictures from the dataset folders

# Following line of code displays output inside a Jupyter notebook
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0


# Now, we will display a batch of 8 dogs and 8 cats pictures:
# Setting up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_dogs_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dogs_names[pic_index-8:pic_index]]
next_cats_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cats_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_dogs_pix+next_cats_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()




# Preprocessing the training set and applying data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(r'C:\Users\mramd\Desktop\Coursera\TensorFlowCNN_Project\Classification\training_set\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')



# Preprocessing the validation set
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = test_datagen.flow_from_directory(r'C:\Users\mramd\Desktop\Coursera\TensorFlowCNN_Project\Classification\validation_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


### Create and Train the Model:

# Initializing the CNN
cnn = tf.keras.models.Sequential()

# Input dimensions of image is 64*64 with 3 bytes color

# Create the first Convolutional Layer.Need only to specify input shape for first layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Create a Pooling Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# This the second Convolutional Layer was removed
#cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# This Pooling Layer was removed
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten the results inorder to feed into the fully connected CNN below
cnn.add(tf.keras.layers.Flatten())

# Create fully Connected Convolutional Neural Network with 128 neuron hidden layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Create Output Layer with a sigmoid function
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))




# Call and display model summary:
cnn.summary()


# We will train our model with the binary_crossentropy loss function, because it's a binary classification problem and our final activation is a sigmoid.
# We will use the 'Adam' optimizer:

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Train the CNN on the training set and evaluate it on the test set
# Juptyer notebook version can crash here sometimes.Try lowering the no. of epochs and number of layers

cnn.fit(x = training_set, validation_data = validation_set, epochs = 5)


### Test the Model and Make Predictions:


# Testing the CNN on input image of an Border Collie
import numpy as np

# Note keras.preprocessing API is deprecated in Tensorflow 2.9.1
# Commented out "from keras.preprocessing import image"

from keras_preprocessing import image

# Enter path to your test image
test_image = image.load_img(r'C:\Users\...\...\...\...\...\border_collie\collie.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)

training_set.class_indices

if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print("This is a ",prediction)

