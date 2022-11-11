import os
import numpy as np
import keras
import pandas as pd
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
from sklearn.model_selection import train_test_split
import shutil
from shutil import unpack_archive
from collections import OrderedDict
from keras_preprocessing import image
#import matplotlib.pyplot as plt
import cv2

class CNN (object):
    def __init__(self, dataset_path, img_width, img_height, batch_size, epochs, num_classes, model_path):
        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.model_path = model_path
        self.label_map = {}

    def train(self):
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        self.read_clean_data()

        train_generator = train_datagen.flow_from_directory(
            self.dataset_path + "output/multi_train",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')
        self.label_map = (train_generator.class_indices)

        validation_generator = test_datagen.flow_from_directory(
            self.dataset_path + "output/multi_val/",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        testing_generator = test_datagen.flow_from_directory(
            self.dataset_path + "output/multi_test/",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'            
        )

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
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


        checkpointer = ModelCheckpoint(filepath=self.model_path, 
                                        verbose=1, 
                                        save_best_only=True, 
                                        monitor='val_loss',
                                        save_freq='epoch')
        model.fit_generator( train_generator, 
                            steps_per_epoch=train_generator.n // self.batch_size, 
                            epochs=self.epochs, 
                            validation_data=validation_generator, 
                            validation_steps=testing_generator.n // self.batch_size, 
                            callbacks=[checkpointer])

        # Save model
        model.save(self.model_path)

    def read_clean_data(self):
        lfw_allnames = pd.read_csv(self.dataset_path + "lfw_allnames.csv")

        image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
        image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
        image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
        image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
        image_paths = image_paths.drop("images",axis=1)

        # Separate classes
        multi_data = pd.concat([image_paths[image_paths.name=="George_W_Bush"].sample(75),
                        image_paths[image_paths.name=="Colin_Powell"].sample(75),
                        image_paths[image_paths.name=="Tony_Blair"].sample(75),
                        image_paths[image_paths.name=="Donald_Rumsfeld"].sample(75),
                        image_paths[image_paths.name=="Gerhard_Schroeder"].sample(75),
                        image_paths[image_paths.name=="Ariel_Sharon"].sample(75)])

        print("Multi_Data ",len(multi_data))

        multi_train, multi_test = train_test_split(multi_data, test_size=0.3)
        multi_train, multi_val = train_test_split(multi_train,test_size=0.3)

        self.directory_mover(multi_train,"multi_train/")
        self.directory_mover(multi_val,"multi_val/")
        self.directory_mover(multi_test,"multi_test/")


    def directory_mover(self,data,dir_name):
        co = 0
        for image in data.image_path:
            # create top directory
            if not os.path.exists(os.path.join(self.dataset_path + 'output/',dir_name)):
                shutil.os.mkdir(os.path.join(self.dataset_path + 'output/',dir_name))

            data_type = data[data['image_path'] == image]['name']
            data_type = str(list(data_type)[0])
            if not os.path.exists(os.path.join(self.dataset_path + 'output/',dir_name,data_type)):
                shutil.os.mkdir(os.path.join(self.dataset_path + 'output/',dir_name,data_type))
            path_from = os.path.join(self.dataset_path + 'lfw-deepfunneled/',image)
            path_to = os.path.join(self.dataset_path + 'output/',dir_name,data_type)
            # print(path_to)
            shutil.copy(path_from, path_to)
            # print('Moved {} to {}'.format(image,path_to))
            co += 1

        print('Moved {} images to {} folder.'.format(co,dir_name))


    def clean(self):
        # Clean image used for testing
        if "multi_train" in os.listdir(self.dataset_path + "output/"):
            shutil.rmtree(self.dataset_path + "output/multi_train")
        if "multi_val" in os.listdir(self.dataset_path + "output/"):
            shutil.rmtree(self.dataset_path + "output/multi_val")
        if "multi_test" in os.listdir(self.dataset_path + "output/"):
            shutil.rmtree(self.dataset_path + "output/multi_test")

    def get_stats(self):
        multi_test_names = []

        test_datagen = ImageDataGenerator(rescale=1./255)
        multi_test_set = test_datagen.flow_from_directory(
            self.dataset_path + "output/multi_test/",
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'            
        )

        for i in range(len(multi_test_set.filenames)):
            multi_test_names.append(multi_test_set.filenames[i])
        for i in range(len(multi_test_names)):
            multi_test_names[i] = multi_test_names[i].split("/")[0]

        multi_test_name_order = list(OrderedDict.fromkeys(multi_test_names))
        for i in range(len(multi_test_name_order)):
            multi_test_name_order[i] = multi_test_name_order[i].replace("\\","/")
            
        predictions_values = []
        predictions_len = []
        for i in range(len(multi_test_name_order)):
            print(multi_test_name_order[i])
        

        m_p0 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[0] + "/")
        m_p1 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[1] + "/")
        m_p2 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[2] + "/")
        m_p3 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[3] + "/")
        m_p4 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[4] + "/")
        m_p5 = self.prediction(self.dataset_path + "output/multi_test/" + multi_test_name_order[5] + "/")

        
        multi_predic_frame = pd.DataFrame(list(zip(m_p0 + m_p1 + m_p2 + m_p3 + m_p4 + m_p5,
                                                [0] * len(m_p0) + [1] * len(m_p1) + [2] * len(m_p2) + [3] * len(m_p3) + [4] * len(m_p4) + [5] * len(m_p5))),
                                       columns = ['Predictions','Actual'])
        stats = self.prec_acc(multi_predic_frame)
        print("Precision: " + str(stats[1]))
        print("Recall: " + str(stats[2]))
        print("Classes: " + str(multi_test_name_order))
        
    def prediction_multi_imgs(self, img_path):
        preds = []
        model = keras.models.load_model(self.model_path)
        for img_name in os.listdir(img_path):
            try:
                img = cv2.imread(img_path + str(img_name))
                print(img_path + str(img_name))
                imag = self.preprocess_img(img)
                x = image.img_to_array(imag)
                x = np.expand_dims(x, axis=0)
                result = np.argmax(model.predict(x))
                preds.append(result)
            except Exception as e:
                print(str(e))
        return preds

    def prediction_single_img(self, img_path):
        model = keras.models.load_model(self.model_path)
        try:
            img = cv2.imread(img_path)
            imag = self.preprocess_img(img)
            x = image.img_to_array(imag)
            x = np.expand_dims(x, axis=0)
            result = np.argmax(model.predict(x))
        except Exception as e:
            print(str(e))
            return 'Não Identificado'
        
        class_label = [k for k, v in label_map.items() if v == result]
        return class_label


    def preprocess_img(self,img):
        cropped_image = self.find_and_crop_face(img)
        border_image = self.resize_with_pad(cropped_image)
        return border_image

    def find_and_crop_face(self, img):
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') 
        detected_faces = face_cascade.detectMultiScale(grayscale_image)

        col = detected_faces[0][0]
        r = detected_faces[0][1]
        w = detected_faces[0][2]
        h = detected_faces[0][3]

        new_row_x = r
        new_row_y = r+w
        new_col_x = col
        new_col_y = col+h
        cropped_image = img[new_row_x:new_row_y, new_col_x:new_col_y]
        return cropped_image
    
    def resize_with_pad(self,img):
        width = img.shape[0]
        height = img.shape[1]
        ratio = float(max([250,250])/max([width,height]))
        new_size = tuple([int(width*ratio), int(height*ratio)])
        try:
            imag = cv2.resize(img, new_size)
        except Exception as e:
            print(str(e))
            print("Size not supported")
            return 
        delta_w = 250 - new_size[0]
        delta_h = 250 - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        border_image = cv2.copyMakeBorder(imag, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))
        return border_image

    def predict_image(self, img_dir):
        p = self.prediction(img_dir)
        p_frame = pd.DataFrame(list(zip(p)))
        print(p_frame)
        

    def prec_acc(self, df):
        precision = []
        accuracy = []
        recall = []
        for i in range(len(set(df.Predictions))):
            tp = df[np.logical_and(df['Actual'] == i, df['Predictions'] == i)].shape[0]
            tn = df[np.logical_and(df['Actual'] != i, df['Predictions'] != i)].shape[0]
            fp = df[np.logical_and(df['Actual'] != i, df['Predictions'] == i)].shape[0]
            fn = df[np.logical_and(df['Actual'] == i, df['Predictions'] != i)].shape[0]
            total_preds = df.shape[0]
            try:
                precision.append(tp/(tp + fp))
                accuracy.append((tp + tn)/total_preds)
                recall.append(tp/(tp + fn))
            except ZeroDivisionError:
                precision = 0 if tp+fp == 0 else precision
                accuracy = 0 if total_preds == 0 else accuracy
                recall = 0 if tp+fn == 0 else recall
        return(accuracy,precision,recall)
