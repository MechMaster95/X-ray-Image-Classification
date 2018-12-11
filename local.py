# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:30:35 2018

@author: Nadim
"""
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

all_xray_df = pd.read_csv(r"H:\CS522_dataSet\Data_Entry_2017.csv")
all_image_paths = {os.path.basename(x): x for x in 
                   glob('H:/CS522_dataSet/images/*.png')}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] =all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.head(5)


#####Reading the images

main_file="images"
path=os.path.join(main_file,"*g")
glob('images/*.png')
os.getcwd()
##since we are talking the first data set
df=all_xray_df
df1=df.iloc[0:4999,:]

###converting the complex labels into binary labels

####Understand better
label_counts = df1['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)

df1['Finding Labels'] = df1['Finding Labels'].map(lambda x: x.replace("No Finding", ''))
##### Chopping up the labels
from itertools import chain
all_labels = np.unique(list(chain(*df1['Finding Labels'].map(lambda x:x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All labels({}): {}'.format(len(all_labels), all_labels))

#####One hot encoding
for c_label in all_labels:
    if len(c_label)>1:
        df1[c_label]=df1['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
df1.sample(3)
        
#######Pruning out cases with less than 200
min_cases=200
all_labels = [c_label for c_label in all_labels if df1[c_label].sum()>min_cases]
print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(df1[c_label].sum())) for c_label in all_labels])

#### Can even out the distribution by assigning weights and creating a new sample
###for later part 

label_counts= 100*np.mean(df1[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')
####Preparing Training Data
#understand better
df1['disease_vec']=df1.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df1, test_size=0.25, random_state=1,stratify=df1["Finding Labels"].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])

##### Creating data generators

from keras.preprocessing.image import ImageDataGenerator
img_size=(128,128)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = False, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

    
def flow_from_dataframe(img_data_gen, in_df,path_col, y_col, **dflow_args):
    base_dir =os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse', **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples=in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ' '
    print("reinserting dataframe: {} images".format(in_df.shape[0]))
    return df_gen

#################################################################################
train_gen = flow_from_dataframe(core_idg, train_df,path_col = "path", y_col="disease_vec", target_size = img_size, 
                                color_mode="grayscale", batch_size=32)

valid_gen = flow_from_dataframe(core_idg, valid_df,path_col = "path", y_col="disease_vec", target_size = img_size, 
                                color_mode="grayscale", batch_size=256)

test_X, test_Y=next(flow_from_dataframe(core_idg, valid_df,path_col='path',y_col='disease_vec',target_size=img_size,color_mode='grayscale',batch_size=1024))

test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = img_size,
                             color_mode = 'grayscale',
                            batch_size = 512)) # one big batch
## Ignore next message from keras, values are r
#################################################################
t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    c_ax.axis('off') 
    
    
##### Building the CNN Model
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPool2D, Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
base_mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
#multi_disease_model.add(Conv2D(1024, kernel_size=(2,2), activation='sigmoid',input_shape =  t_x.shape[1:]))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
#multi_disease_model.add(GlobalAveragePooling2D())
#multi_disease_model.add(Dropout(0.6))

#multi_disease_model.add(Dense(1024))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                           metrics = ['categorical_accuracy'])
multi_disease_model.summary()   
###Weights load 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]



history = multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 1, 
                                  callbacks = callbacks_list)

###checking 
score = multi_disease_model.evaluate(test_X, test_Y, verbose=0) 
print("Accuracy after one epoch: " + str(round(score[1]*100,2)) + "%")

pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')

