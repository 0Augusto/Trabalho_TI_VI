
import keras
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

class CNN (object):
    def __init__(self, train_path, test_path, img_width, img_height, batch_size, epochs, num_classes, model_path):
        self.train_path = train_path
        self.test_path = test_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.model_path = model_path

    def train(self):
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        # Model
        model = Sequential()
        model.add(Conv2D(96, (7, 7), strides=(2, 2), padding='same', input_shape=(self.img_width, self.img_height, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, (1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        #modelo de treinamento reconhecimento de faces usando o modelo de treinamento do AlexNet
        model.add(Dense(self.num_classes, activation='softmax'))
        
        #model.summary()
         
        # Training
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=self.model_path, verbose=1, save_best_only=True)
        model.fit_generator( train_generator, steps_per_epoch=2000 // self.batch_size, epochs=self.epochs, validation_data=validation_generator, validation_steps=800 // self.batch_size, callbacks=[checkpointer])
         
        # Save model
        model.save(self.model_path)
        
    def predict(self, img_path):
        model = keras.models.load_model(self.model_path)
        img = image.load_img(img_path, target_size=(self.img_width, self.img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        return preds
    