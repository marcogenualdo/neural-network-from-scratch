import numpy as np
import struct as st
from random import shuffle 
import json


class Brain:
    def __init__ (self, weight_Mat, biases):
        self.nLayers = len(biases)

        self.wMat = weight_Mat 
        self.bias = biases

            
    def read (self,v0):
        result = [v0] #result[k] = v[k] = W[k-1]*sigma(v[k-1])+b[k-1]
        appo = v0 #appo = sigma(result[k]) = sigma(v[k]) 
        
        for k in range(self.nLayers):
            result.append(self.wMat[k].dot(appo) + self.bias[k])
            appo = sigma(result[k+1])
        return result

    
    def guess (self,x):
        for w,b in zip(self.wMat,self.bias):
            x = sigma (w.dot(x) + b)
        return np.argmax(x)

        
    def SGD_training (self, data, eta = 0.5, size_batch = 10, 
            regularize = 0, gamma = 0.5):
        #initializing gradients
        avg_dW = [np.zeros(np.shape(it)) for it in self.wMat]
        avg_db = [np.zeros(len(it)) for it in self.bias]

        #training
        len_data = len(data)
        cost = 0
        for i in range(len_data):
            #computing gradients
            result = self.read (data[i][0])
            dW, db = self.gradErr (result, data[i][1]) 
            cost += C(sigma(result[-1]), data[i][1])

            #adding to gradients' averages
            for j in range(self.nLayers):
                avg_dW[j] += dW[j]
                avg_db[j] += db[j]

            if i % size_batch == 0 and i > 1:
                for j in range(self.nLayers):
                    #regularizing if requested
                    if regularize == 1:
                        self.wMat[j] -= eta * gamma / len_data * np.sign (self.wMat[j])
                    if regularize == 2:
                        self.wMat[j] *= 1 - eta * gamma / len_data 
                    
                    #computing new weights and biases
                    self.wMat[j] -= (eta / size_batch) * avg_dW[j]
                    self.bias[j] -= (eta / size_batch) * avg_db[j]

                    avg_dW[j] = np.zeros(np.shape(self.wMat[j]))
                    avg_db[j] = np.zeros(np.shape(self.bias[j]))
        
        return cost / len_data
       

    def gradErr (self, v, sol):
        dE = dC(sigma(v[-1]),sol) #* dsigma(v[-1])
        Wgrad = [np.tensordot (dE,sigma(v[-2]), axes=0)]
        bgrad = [dE]

        for k in range (1, self.nLayers):
            dE = dE.dot (self.wMat[-k]) * dsigma(v[-k-1])
            Wgrad.append (np.tensordot (dE,sigma(v[-k-2]), axes=0))
            bgrad.append (dE)
        
        Wgrad.reverse()
        bgrad.reverse()
        return Wgrad, bgrad


    def save_to_json (self, file_name = 'network2.json'):
        data = {"weights" : [w.tolist() for w in self.wMat],
                "biases" : [b.tolist() for b in self.bias]}
        
        outf = open (file_name, 'w') 
        json.dump(data, outf)
        outf.close()
        

#sigmoid function and its derivative
#sigma = lambda x: np.arctan(x) / np.pi + 0.5
#dsigma = lambda x: 1 / (1 + x*x) / np.pi

sigma = lambda z: 1 / (1 + np.exp(-z))
dsigma = lambda z: sigma(z) * (1 - sigma(z))

#cost function derivative
#C = lambda x,a: 1/2 * np.sum ((x - a) * (x - a))
dC = lambda x,a: x - a 

C = lambda x,a: - np.sum (a * np.log(x) + (1 - a) * np.log(1 - x))
#do not use the following derivative for cross entropy, use dC above and comment
#'*dsigma(v[-1])' in the gradErr function
#dC = lambda x,a: np.array ([(xi - zi) / (xi * (1 - xi)) for xi, zi in zip(x,a)]) 


def test (network, data):
    errors = 0
    for image, sol in data:    
        g = network.guess (image)
        if g - sol:
            errors += 1

    return errors


def load_from_json (file_name):
    with open(file_name, 'r') as fin:
        data = json.load(fin)

        weights = [np.array(w) for w in data["weights"]]
        biases = [np.array(b) for b in data["biases"]]

        return Brain(weights, biases)


def one_hot (numbers, dim):
    L = len(numbers)
    vec = np.zeros((np.size(numbers), dim))

    for k in range(L):
        vec[k][numbers[k]] = 1;
    return vec


def speaker (f):
    def wrapper (*args,**kwargs):
        print('Loading data...', end = '')
        a = f(*args,**kwargs)
        print('Done.')
        return a
    return wrapper


@speaker
def load_images (img_name, lbl_name, load_up_to = 'all', vectorize = 0):
    img_file = open(img_name, 'rb')
    lbl_file = open(lbl_name, 'rb')

    img_file.seek(0) #getting to the beginning of the file
    lbl_file.seek(8) #labels start from the 8-th byte of this file
    magic_n = st.unpack('>4B', img_file.read(4)) #magic number

    nImgs = st.unpack('>I',img_file.read(4))[0] #number of images
    nRows = st.unpack('>I',img_file.read(4))[0] #number of rows
    nCols = st.unpack('>I',img_file.read(4))[0] #number of columns
    if load_up_to != 'all':
        nImgs = int(load_up_to)

    #array of images
    images = np.zeros((nImgs,nRows*nCols))
    images =  np.asarray(st.unpack('>'+'B'*nRows*nCols*nImgs,
    img_file.read(nImgs*nRows*nCols))).reshape((nImgs,nRows*nCols)) / 255

    #array of labels
    labels = np.asarray(st.unpack('>'+str(nImgs) + 'B',lbl_file.read(nImgs))).reshape(nImgs)
    if vectorize: 
        labels = one_hot (labels, vectorize)

    img_file.close()
    lbl_file.close()
    return list(zip(images, labels))


def std_exec ():
    #neural network layers
    dims = [784, 100, 10]
    #intializing weights and biases at random
    weights, biases = [], []
    for i in range(len(dims)-1):
        weights.append(0.1 * np.random.randn(dims[i+1],dims[i]))
        biases.append(0.1 * np.random.randn(dims[i+1])) 

    network = Brain(weights,biases)

    #loading datasets
    training_files = ('mnist/data/train-images-idx3-ubyte','mnist/data/train-labels-idx1-ubyte') 
    test_files = ('mnist/data/t10k-images-idx3-ubyte','mnist/data/t10k-labels-idx1-ubyte')
    
    training_data = load_images (*training_files, vectorize = 10)
    test_data = load_images (*test_files)

    #training and testing
    reg = 0 
    for i in range(30):
        shuffle (training_data)
        cost = network.SGD_training (training_data, eta = 0.6)
        errors = test (network, test_data)

        perc = (1 - errors / len(test_data)) * 100
        print('Epoch {0} results: {1} mistakes, {2:.2f}% correct, training cost = {3:.6f}'
                .format(i + 1, errors, perc, cost))
        
    network.save_to_json('network2.json')


if __name__ == '__main__':
    std_exec()
