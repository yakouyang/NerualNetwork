# -*- coding: UTF-8 -*-

import numpy as np
from activators import Sigmoid
from data_loader import *
from datetime import datetime
import pickle

class DenseLayer(object):
    def __init__(self,input_size,output_size,activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        #Initialization weight W, offset term b, and output vector
        self.W = np.random.uniform(-0.1,0.1,(output_size,input_size))
        self.b = np.zeros((output_size,1))
        self.output = np.zeros((output_size,1))
        
    def feed_forward(self,input_vector):
        self.input = input_vector
        #activator.function is a function of the activation function for forward calculation
        self.output = self.activator.function(np.dot(self.W,input_vector)+self.b)
    
    def back_propagation(self,delta_vector):
        #activator.deviation Is the activation function derivation function for backpropagation calculation
        self.delta = self.activator.deviation(self.input)*np.dot(self.W.T,delta_vector)
        self.W_gradient = np.dot(delta_vector,self.input.T)
        self.b_gradient = delta_vector
    
    def update(self, learning_rate):
        self.W += learning_rate * self.W_gradient
        self.b += learning_rate * self.b_gradient
        
    def show_parameter(self):
        print('Weights:\n%s\nbiases:\n%s' % (self.W, self.b))
        
class Network(object):
    def __init__(self, layers):
        self.layers= []
        for i in range(len(layers)-1):
            self.layers.append(
                DenseLayer(layers[i],layers[i+1],Sigmoid())
            )
    
    def predict(self,sample):
        output = sample
        for layer in self.layers:
            layer.feed_forward(output)
            output = layer.output
        return output
        
    def calc_gradient(self,label):
        # When calculating the gradient, the residual propagates from back to front
        delta = self.layers[-1].activator.deviation(self.layers[-1].output)*(label-self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.back_propagation(delta)
            delta = layer.delta
        return delta
    
    def update_weight(self,learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
        
    def train(self,data_set,labels,learning_rate,epoch,early_stopping,patience,show_verbose):
        best_accuracy = -1
        stopping_step = 0
        for i in range(epoch):
            data_set,labels = self.shuffle_data(data_set,labels)
            for index in range(len(data_set)):                
                self.train_single_sample(data_set[index],labels[index],learning_rate)
            current_accuracy = self.evaluate(data_set,labels)
            if show_verbose:
                print('%s epoch %d finished, loss: %f accuracy: %f' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),i,self.loss(self.predict(data_set[-1]),labels[-1]),current_accuracy))
            if early_stopping == True:
                if current_accuracy > best_accuracy:
                    stopping_step = 0
                    best_accuracy = current_accuracy
                else:
                    stopping_step += 1
                if  stopping_step >= patience:
                    print("early stopping!")
                    break
    
    def train_single_sample(self,sample,label,learning_rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(learning_rate)
    
    def shuffle_data(self,data_set,labels):
        s = np.arange(len(data_set))
        np.random.shuffle(s)
        data_set = data_set[s]
        labels = labels[s]
        return data_set,labels
               
    def show_parameter(self):
        for layer in self.layers:
            layer.show_parameter()
    
    def loss(self,output,label):
        return 0.5*((label-output)**2).sum()
    
    def get_result(self,one_hot_vector):
        return np.argwhere(one_hot_vector == one_hot_vector.max())[0][0]
        
    def evaluate(self,data_set,labels):
        # The assessment result is accuracy
        correct = 0
        total = len(data_set)
        
        for i in range(len(data_set)):
            predict = self.get_result(self.predict(data_set[i]))
            expected = self.get_result(labels[i])
            if predict == expected:
                correct += 1
        return float(correct) / total
        
    def check_gradient(self,sample,label,epsilon):
        self.predict(sample)
        self.calc_gradient(label)
        
        for layer in self.layers:
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    origin_W = layer.W[i,j]
                    layer.W[i,j] += epsilon
                    output = self.predict(sample)
                    L1 = self.loss(output,label)
                    layer.W[i,j] = origin_W
                    layer.W[i,j] -= epsilon
                    output = self.predict(sample)
                    L2 = self.loss(output,label)
                    calc_gradient = -(L1 - L2) / (2 * epsilon)
                    layer.W[i,j] += epsilon
                    if calc_gradient > epsilon :
                        print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                            i, j, calc_gradient, layer.W_gradient[i,j]))


def check_gradient(x_train,y_train,layers=[784,10,10],epsilon = 10e-8):
    '''
    Gradient check
    '''
    data_set = x_train 
    labels = y_train
    network = Network(layers)
    network.check_gradient(data_set[0],labels[0],epsilon)
    return network

def data_reshape(args):
    '''
    Turn each 2D array into 3D, such as changing the shape of x_train from (60000,784) to (60000,784,1)
    Each sample is transformed from a row vector to a column vector
    Necessary processing, otherwise there will be a problem of dimensional errors in the vector operations inside the network
    '''
    return map(
        lambda arg: arg.reshape([arg.shape[0],arg.shape[1],1])
        , args
    )

def random_visualized_test(network,x_test,y_test,data_loader,numbers):
    '''
    Realize randomly selected image display on the test set, 
    and give the real results of the image and the corresponding network prediction results
    '''
    for i in range(numbers):
        random_index = np.random.randint(0,len(x_test))
        predict = network.get_result(network.predict(x_test[random_index]))
        expected = network.get_result(y_test[random_index])
        print("real: %d - predict: %d" % (expected,predict))
        print("pic_index: %d" % random_index )
        data_loader.show("x_test",random_index)

def save_model(network,filename="trained_model.pickle"):
    '''
    Save the trained network
    '''
    with open(filename, "wb") as file_:
        pickle.dump(network, file_, -1)
        print("The trained model is saved successfully!")

def load_model(filename="trained_model.pickle"):
    '''
    Load a trained network from a file
    '''
    network = pickle.load(open(filename, "rb", -1))
    return network

def fetch_data():
    '''
    Get mnist data
    '''
    #data_reshape is necessary to convert the row vector to a column vector (the entire source code uniform style), otherwise the network internal calculation will have a problem of dimension matching
    x_train,y_train,x_test,y_test = data_reshape(get_whole_data())
    data_loader = DataLoader('mnist.npz')
    return x_train,y_train,x_test,y_test,data_loader
    
def build_model(x_train,y_train,layers=[784, 250, 10],learning_rate=0.01,epoch = 50,save_flag=True,early_stopping=True,patience=5,show_verbose=True):
    '''
    Create a network and train
    '''
    network = Network(layers)
    network.train(x_train,y_train, learning_rate, epoch,early_stopping,patience,show_verbose)
    if save_flag ==True:
        save_model(network)
    return network

def evaluate_model(network,x_test,y_test,data_loader,show_pic_number=10):
    '''
    Evaluate the network's forecasting effect on the test set and randomly select pictures for visual display
    '''
    print("x_test accuracy: %f" % network.evaluate(x_test,y_test))
    random_visualized_test(network,x_test,y_test,data_loader,show_pic_number)

if __name__ == '__main__':
    x_train,y_train,x_test,y_test,data_loader=fetch_data()
    
    '''
    #Part 1: Gradient check
    check_gradient(x_train,y_train)
    '''
    '''
    #Part 2: Retrain a network
    network = build_model(x_train,y_train,layers=[784, 250, 10],epoch=50)
    evaluate_model(network,x_test,y_test,data_loader,show_pic_number=10)
    '''
    
    #Part 3: Load an already trained network for forecasting performance evaluation
    network = load_model()
    evaluate_model(network,x_test,y_test,data_loader,show_pic_number=10)
    
