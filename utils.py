#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[2]:


def unpickle(file):
    "Load cifar-10 data"
    
    with open(file,"rb") as file:
        data = pickle.load(file,encoding="bytes")
    return data


# In[3]:


def one_hot_encoded(labels):
    """
    One hot encoding of label (Y)
    
    
    Returns: (m, length(set(labels)))
    """
    size = len(labels)
    output_one_hot = np.zeros((10,size))
    for label in range(len(labels)):
        output_one_hot[labels[label],label] = 1 
    return output_one_hot


# In[4]:


def word_label(Y, label_names):
    """
    This function transforms one hot encoded Y values to word labels
    
    Arguments:
    
    (Y)- One hot encoded labels of Y
    
    Return:
    (word_category_matrix)- Word categories of Y
    """
    m = Y.shape[1]
    word_category_matrix = []
    word_categories = label_names
    position = np.argmax(Y,axis = 0)

    for i in range(0,m):
        word_category_matrix.append(word_categories[position[i]])
    return word_category_matrix


# In[5]:


def load_cifar_10_data():
    """
    Returns files, data and labels for training and test sets
    """
    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: 3072  
    # no_of_batches: 6
    
    # initialize training data
    cifar_train_data = None
    cifar_train_labels = []
    cifar_train_label_names = []
    data_dir = 'cifar-10-batches-py' #cifar-10 dataset directory
    
    # metadata about cifar-10 dataset
    cifar_train_metadata = unpickle(data_dir + "/batches.meta")
    cifar_train_label_namelist = cifar_train_metadata[b'label_names']
    
    for i in range(1,6):
        # Obtain cifar training datasets
        
        cifar_train_dict = unpickle(data_dir + "\data_batch_" + str(i))
        cifar_train_data_temp = np.asarray(cifar_train_dict[b"data"])
        cifar_train_data_temp = cifar_train_data_temp.reshape(cifar_train_data_temp.shape[0],3,32,32).transpose(0,2,3,1)
        cifar_train_labels_temp = np.asarray(cifar_train_dict[b"labels"])


        #Concatenate the datasets to form entire dataset
        if i == 1:
            cifar_train_data = cifar_train_data_temp
            cifar_train_labels = cifar_train_labels_temp
        else:
            cifar_train_data = np.concatenate((cifar_train_data,cifar_train_data_temp),axis = 0)
            cifar_train_labels = np.concatenate((cifar_train_labels,cifar_train_labels_temp), axis = -1)
            
    cifar_train_data = cifar_train_data/255
    cifar_train_labels = one_hot_encoded(cifar_train_labels)
    cifar_train_label_names = word_label(cifar_train_labels, cifar_train_label_namelist)
    
    #Reshape the vector from 3702 dimensions to mx32x32x3
    return cifar_train_data, cifar_train_labels, cifar_train_label_names
      


# In[10]:


def show_data():
    cifar_10_dir = 'cifar-10-batches-py'

    train_data, train_labels, train_label_names = load_cifar_10_data()

    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    train_data = train_data
    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()


# In[11]:





# In[ ]:




