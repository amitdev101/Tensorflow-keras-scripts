import tensorflow as tf 
print(tf.__version__)
# from tensorflow import keras
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
import os


print("import successful")


training_path = "./dataset/training_set"
print(training_path)

################# Defining Model ########################

model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=2,activation='softmax',name='output')) # 2 units because of 2 classes for activation softmax
#                                                            
    
################## Compiling Model ##########################
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])

################## Training the Model ##################
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,) 

test_datagen = ImageDataGenerator(rescale = 1./255,)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                        target_size = (64, 64), 
                                                        batch_size = 32,
                                                        class_mode = 'categorical'
                                                    )
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32, 
                                                    class_mode = 'categorical'
                                                    )

history = model.fit_generator(training_set, 
                            epochs = 10,
                            validation_data = test_set,
                            )


######################## Saving the Model ##############

model.save('mymodel.h5')          


#################### Predicting a new image with out model ##############
