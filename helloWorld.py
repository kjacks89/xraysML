from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers 
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# dimensions of our images
img_width, img_height = 299, 299

# directory and image information
train_data_dir = '/home/kyle/emory/ml/chestorab/data/train'
validation_data_dir = '/home/kyle/emory/ml/chestorab/data/val'

train_samples = 65
validation_samples = 10
# epochs = number of passes through training data
epochs = 20
# batch_size = number of images processed at same time
batch_size = 5

# build the Inception V3 network, use pretrained weights from ImageNet
# remove top fully connected layers by include_top = False
base_model = applications.InceptionV3(weights='imagenet', include_top = False,
        input_shape = (img_width, img_height, 3))

# build a classifier model to put on top of the convolutional model
# This consists of a global average pooling layer and a fully connected layer
# with 256 nodes.  Then apply dropout and sigmoid activation
model_top = Sequential()
model_top.add(GlobalAveragePooling2D(input_shape = base_model.output_shape[1:],
    data_format = None))
model_top.add(Dense(256, activation = 'relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1, activation = 'sigmoid'))
model = Model(inputs = base_model.input, outputs = model_top(base_model.output))

# Compile model using Adam optimizer with common values and binary cross entropy loss
# Use low learning rate (lr) for transfer learning
model.compile(optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-08, decay = 0.0), loss = 'binary_crossentropy',
    metrics = ['accuracy'])

# Some on-the-fly augmentation options
train_datagen = ImageDataGenerator(
        rescale = 1./255, # Rescale pixel values to 0-1 to aid CNN processing
        shear_range = 0.2, # 0-1 range for shearing
        zoom_range = 0.2, # 0-1 range for zoom
        rotation_range = 20, # 0-180 range, degrees of rotation
        width_shift_range = 0.2, # 0-1 range for horizontal translation
        height_shift_range = 0.2, # 0-1 range vertical translation
        horizontal_flip = True) # set True or False

val_datagen = ImageDataGenerator(
        rescale = 1./255) # Rescale pixel values to 0-1 to aid CNN processing

# Directory, image size, batch size already specified above
# Class mode is set to 'binary' for a 2-class problem
# Generator randomly shuffles and presents images in batches to the network
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        class_mode = 'binary')

validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size, 
        class_mode = 'binary')

# Fine-tune the pretrained Inception V3 model using the data generator
# Specify steps per epoch (number of samples/batch_size)

history = model.fit_generator(
        train_generator, 
        steps_per_epoch = train_samples // batch_size,
        epochs = epochs, 
        validation_data = validation_generator, 
        validation_steps = validation_samples // batch_size)

