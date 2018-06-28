# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self,path):
        print("mnist file path: %s"%path)
        self.path = path
        self.get_file_content()
        
    def get_file_content(self):
        f = np.load(self.path)
        self.x_train, self.y_train = f['x_train'], f['y_train']
        self.x_test, self.y_test = f['x_test'], f['y_test']        
        f.close()
        
    def show(self,choose,index):
        '''
        Show a picture, usage is as follows：
        data_loader = DataLoader('mnist.npz')
        data_loader.show("train",34567)
        '''
        if choose == "train":
            dataset = self.x_train
        else:
            dataset = self.x_test
        im = dataset[index,:,:]
        plt.imshow(im, cmap='binary')
        plt.show()
        
    def label_one_hot(self,y):
        label_vectors = []
        for y_label in y:
            label_vec = []
            for i in range(10):
                if i == y_label:
                    label_vec.append(1)
                else:
                    label_vec.append(0)
            label_vectors.append(label_vec)
        label_vectors = np.array(label_vectors) 
        return  label_vectors
    
    def load(self,normalize=True):
        '''
        Convert each picture to a one-dimensional vector and convert each tag number to one-hot vector
        '''
        x_train = self.x_train.reshape([60000,784]).astype(np.float)
        x_test = self.x_test.reshape([10000,784]).astype(np.float)
        y_train = self.label_one_hot(self.y_train)
        y_test = self.label_one_hot(self.y_test)
        if normalize == True:
            x_train /= 255
            x_test /=255
        print("x_train size: ",x_train.shape)
        print("y_train size: ",y_train.shape)
        print("x_test size: ",x_test.shape)
        print("y_test size: ",y_test.shape)
        return x_train,y_train,x_test,y_test
        
def get_whole_data(normalize=True):
    data_loader = DataLoader('mnist.npz')
    return data_loader.load(normalize)

#X_train,Y_train,X_test,Y_test = get_whole_data()





