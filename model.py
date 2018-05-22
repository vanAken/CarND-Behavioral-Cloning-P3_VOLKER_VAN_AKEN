import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D,Lambda,BatchNormalization,Dropout,Flatten,Dense,Activation
from keras.layers.convolutional  import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2


def csv_reader(paths): 
    '''
    read the location of the images and driving data form csv file and return a list of this data 
    '''
    #inilize variables
    samples = []
    last_len_sampel =0
    for path in paths:
        with open(path+'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        print(path,'  samples:',len(samples)-last_len_sampel)
        last_len_sampel=len(samples)        
    # each sample will result in 6 images, because the left and right images will be flipped 3*2=6
    print(len(samples),'samples results in ',len(6*samples),' images.')
    return samples

def plotting_histogram(samples):
    '''
    visualize the statistical information of the input data with gauss curve
    '''
    angles = []
    for angle_sample in samples:
        angle = float(angle_sample[3])
        angles.append(angle)
 
    # number of bins for sorting the input
    bins = 31
    # parameter for the Gauss curve 
    mu    = np.mean(angles)
    sigma = np.std(angles)
    # the histogram of the data
    n, bins, patches = plt.hist(angles, bins, density=True, facecolor='green', alpha=0.75)
    gauss = mlab.normpdf( bins, mu, sigma)    
    plt.plot(bins, gauss, 'r--', linewidth=2)
    #plot
    plt.xlabel('Steering angles [-]')
    plt.ylabel('Probability [-]')
    plt.title('Histogram of {} steering angles'.format(len(samples)))
    plt.grid(True)
    plt.show()

def process_image(img):
    '''
    Convert color representation from red/green/blue to blue/green/red 
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #use a bilateral Filter, this help to find the edges for the CNN - NO!
    #kernel_size = 15
    #img = cv2.bilateralFilter(img,kernel_size,150,150)
    return img 

def generator(samples, batch_size, correction):
    '''
    use a generator to save memory 
    '''
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                center_angle= float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                left__angle = center_angle + correction
                right_angle = center_angle - correction
                                
                # read in images from center, left and right cameras
                center_image= process_image(np.asarray(Image.open(batch_sample[0]))) 
                left__image = process_image(np.asarray(Image.open(batch_sample[1])))
                right_image = process_image(np.asarray(Image.open(batch_sample[2])))
                #img = cv2.rectangle(center_image, (0,28), (320,132), (0,255,0), 2)
                #cv2.imwrite(batch_sample[0]+'_bila.png',img)
 
                # extend angles and -angles ; images / flip to data set
                angles.append(center_angle)
                #angles.extend((center_angle,-center_angle))                
                angles.extend((+center_angle,+left__angle,+right_angle))
                angles.extend((-center_angle,-left__angle,-right_angle))
                
                images.append(center_image) 
                #images.extend((center_image,cv2.flip(center_image,1)))
                images.extend((center_image,left__image,right_image))
                images.extend((cv2.flip(center_image,1),cv2.flip(left__image,1),cv2.flip(right_image,1)))
      
            # convert the list in np.array for Keras
            X_train, y_train = np.array(images),np.array(angles)
            yield shuffle(X_train, y_train)

def nvidia_net(samples,epochs,correction,test_size=0.2,batch_size=32):

    #split the data
    train_samples, valid_samples = train_test_split(samples, test_size=test_size)
 
    # feed the generator with X
    train_generator = generator(train_samples, batch_size, correction)
    valid_generator = generator(valid_samples, batch_size, correction) 

    # Nvidias Dave-2 mainly modify the strides to fit given the input of 68x320.
    # this reslts in a slightly bigger CNN than Dave-2
    model = Sequential()
    model.add(Cropping2D(cropping=((30,30),(0,0)), input_shape=(160,320,3)))                #3@100x320
    model.add(Lambda(lambda x: x / 255.-.5))                           
    print(model.input_shape)
    print(model.output_shape)

    model.add(Convolution2D(24, (5, 5), padding='valid', kernel_regularizer=l2 (0.0001), activation='elu', strides=(3,3))) #24@32x106
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Convolution2D(32, (5, 5), padding='valid', kernel_regularizer=l2 (0.0001), activation='elu', strides=(2,2))) #36@14x51
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Convolution2D(48, (5, 5), padding='valid', kernel_regularizer=l2 (0.0001), activation='elu', strides=(2,2))) #48@5x24
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Convolution2D(64, (3, 3), padding='valid', kernel_regularizer=l2 (0.0001), activation='elu', strides=(1,1))) #64@3x22
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Convolution2D(64, (3, 3), padding='valid', kernel_regularizer=l2 (0.0001), activation='elu', strides=(1,1))) #64@1x20
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Flatten())
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    print(model.output_shape)

    model.add(Dense(100, kernel_regularizer=l2 (0.0001), activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense( 50, kernel_regularizer=l2 (0.0001), activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense( 10, kernel_regularizer=l2 (0.0001), activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,
                     steps_per_epoch=len(train_samples)/batch_size,
                     validation_data=valid_generator, 
                     validation_steps=len(valid_samples)/batch_size,
                     epochs=epochs)
    #save trained CNN data
    model.save('model.h5')
    print('model saved')
    return history_object
 
def plot_history(history_object):    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
   
##########################################################################################
# Startline
##########################################################################################
my_paths =('./data1/','./data2/','./data3/','./data4/','./UDACity_data/')
my_paths += ('./data_M/','./data_MS/','./data_MB/') 
# locate the images in a list
samples = csv_reader(my_paths)

# statistic plot
plotting_histogram(samples)

# compile and train the model using the generator function
# hyperparameter correction for steering angle
history_object = nvidia_net(samples,epochs=25,correction= .4)

plot_history(history_object)


