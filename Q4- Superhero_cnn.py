
# coding: utf-8

# ### Loading data from clouderizer drive

# In[35]:


get_ipython().system(u'ln -s ../data/ ./')
get_ipython().system(u'ln -s ../out/ ./')


# ### Importing useful library

# In[1]:


import os
import pandas as pd
import numpy as np
from keras.preprocessing import image


# In[5]:


# Convolutional Neural Network


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 12, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)



# ### Checking the indices for each class 

# In[6]:


training_set.class_indices


# ### Checking the files in test folder 

# In[39]:


import os

path, dirs, files = next(os.walk("data/test_images/CAX_Superhero_Test"))
file_count = len(files)
file_count


# ### Creating dataframe to store output for test folder

# In[2]:


df= pd.DataFrame(columns=['filename','Superhero'])


# ### prediction

# In[42]:


i=0
for img in os.listdir('data/test_images/CAX_Superhero_Test'):
    print(img)
    if img != '.DS_Store':
        test_image = image.load_img('data/test_images/CAX_Superhero_Test/'+str(img), target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        print(result)
        for ind,x in enumerate(np.nditer(result)):
            if x==1.0:
                prediction = list(training_set.class_indices.keys())[list(training_set.class_indices.values()).index(ind)]
        df.loc[i] = [img,prediction]
        i+=1
        print(prediction)

print (i)


# In[43]:


df


# ### saving into a csv file 

# In[44]:


df.to_csv('Superhero_3375_SampleSubmission.csv',sep=',',index=False)

