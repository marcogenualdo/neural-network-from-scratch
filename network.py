import numpy as np
import struct as st
import random

class Brain:
    def __init__ (self, weight_Mat, biases):
        self.nLayers = len(biases)

        self.wMat = []
        self.bias = []
        for k in range(self.nLayers):
            self.wMat.append(np.asarray(weight_Mat[k]))
            self.bias.append(np.asarray(biases[k]))

            
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

        
    def train (self, data, gamma = 3.0, size_batch = 10):
        #initializing gradients
        avg_dW = [np.zeros(np.shape(it)) for it in self.wMat]
        avg_db = [np.zeros(len(it)) for it in self.bias]

        #training
        for i in range(len(data)):
            result = self.read (data[i][0])

            #computing gradients
            sol = np.zeros(10)
            sol[data[i][1]] = 1.0
            dW,db = gradErr(self.wMat,self.bias,result,sol,self.nLayers)
            
            #adding to gradients' averages
            for j in range(self.nLayers):
                avg_dW[j] += dW[j]
                avg_db[j] += db[j]

            if i % size_batch == 0 and i > 1:
                for j in range(self.nLayers):
                    #computing new weights and biases
                    self.wMat[j] -= (gamma / size_batch) * avg_dW[j]
                    self.bias[j] -= (gamma / size_batch) * avg_db[j]

                    avg_dW[j] = np.zeros(np.shape(self.wMat[j]))
                    avg_db[j] = np.zeros(len(self.bias[j]))
       

    def save_to_dat (self, file_name = '../saved_models/network.dat'):
        outf = open (file_name, 'w')

        for w,b in zip(self.wMat, self.bias):
            sw = np.shape(w)
            outf.write (str(sw[0]) + ' ' + str(sw[1]) + '\n')    

            for i in range(sw[0]):
                for j in range(sw[1]):
                    outf.write (str(w[i][j]) + ' ')
                outf.write ('\n')

            for i in range(sw[0]):
                outf.write (str(b[i]) + ' ')
            outf.write ('\n')

        outf.close()
        

#sigmoid function and its derivative
sigma = lambda x: np.arctan(x) / np.pi + 0.5
dSigma = lambda x: 1 / (1 + x*x) / np.pi
#sigma = lambda z: 1/(1 + np.exp(-z))
#dSigma = lambda z: sigma(z) * (1 - sigma(z))

#cost function derivative
#C(x,a) = 1/2*(x-a)^2
dC = lambda x,a: x-a 

#gradient evaluator
def gradErr (W,b,v,sol,l):
    dE = dC(sigma(v[l]), sol) * dSigma(v[l])
    Wgrad = [np.tensordot (dE,sigma(v[l-1]), axes=0)]
    bgrad = [dE]
    
    for k in range(1,l):
        dE = dE.dot (W[l-k]) * dSigma(v[l-k])
        Wgrad.append (np.tensordot (dE,sigma(v[l-k-1]), axes=0))
        bgrad.append (dE)

    Wgrad.reverse ()
    bgrad.reverse ()
    return Wgrad, bgrad


def test (network, data):
    errors = 0
    for image, sol in data:
        g = network.guess (image)
        if g - sol:
            errors += 1

    return errors


def load_from_dat (file_name):
    #reading from file
    fin = open (file_name, 'r')
    data = fin.read ()
    fin.close()
    L = len(data)

    #initializing lists
    mlist = []
    vlist = []
    k = 0
    while k < L:
        #reading size
        size = [0, 0]
        for i in range(2):
            size[i], k = build_number (data, k, L)            
            size[i] = int(size[i])

        mat = np.zeros(tuple(size))
        vec = np.zeros(size[0])

        #building matrix
        for i in range(size[0]):
            for j in range(size[1]):
                mat[i][j], k = build_number (data, k, L)
        
        #building vector
        for i in range(size[0]):
            vec[i], k = build_number (data, k, L)
        
        mlist.append (mat)
        vlist.append (vec)

    return Brain (mlist, vlist)


def build_number (data,k,L):
    number = ''
    while data[k] != ' ' and data[k] != '\n':
        number += data[k]
        k += 1

    #skipping spaces and newlines
    while k < L and (data[k] == ' ' or data[k] == '\n'):
        k += 1
    
    return float(number), k




def speaker (f):
    def wrapper (*args,**kwargs):
        if f.__name__ == 'load_images':
            print('Loading data...', end = '')
        a = f(*args,**kwargs)
        print('Done.')
        return a
    return wrapper


#NOTE: n = # of images to be put in the 3-tensor 'images'
@speaker
def load_images (img_name, lbl_name, n = 'all'):
    img_file = open(img_name, 'rb')
    lbl_file = open(lbl_name, 'rb')

    img_file.seek(0) #getting to the beginning of the file
    lbl_file.seek(8) #labels start from the 8-th byte of this file
    magic_n = st.unpack('>4B', img_file.read(4)) #magic number

    nImgs = st.unpack('>I',img_file.read(4))[0] #number of images
    nRows = st.unpack('>I',img_file.read(4))[0] #number of rows
    nCols = st.unpack('>I',img_file.read(4))[0] #number of columns
    if n == 'all': n = nImgs
    
    #array of images
    images = np.zeros((n,nRows*nCols))
    images =  np.asarray(st.unpack('>'+'B'*nRows*nCols*n,img_file.read(n*nRows*nCols))).reshape((n,nRows*nCols)) / 255

    #array of labels
    labels = np.asarray(st.unpack('>'+str(n) + 'B',lbl_file.read(n))).reshape(n)

    img_file.close()
    lbl_file.close()
    return list(zip(images, labels))


def std_exec ():
    training_files = ('../mnist/data/train-images-idx3-ubyte','../mnist/data/train-labels-idx1-ubyte') 
    test_files = ('../mnist/data/t10k-images-idx3-ubyte','../mnist/data/t10k-labels-idx1-ubyte')
    #neural network layers
    dims = [784, 100, 10]
    #intializing weights and biases at random
    weights, biases = [],[]
    for i in range(len(dims)-1):
        weights.append(2*np.random.rand(dims[i+1],dims[i])-1)
        biases.append(2*np.random.rand(dims[i+1])-1) 

    network = Brain(weights,biases)

    #loading datasets
    training_data = load_images (*training_files)
    test_data = load_images (*test_files)


    #training and testing
    for i in range(30):
        random.shuffle(training_data)
        network.train(training_data)
        errors = test(network, test_data)
        perc = (1-errors/len(test_data))*100
        print(i+1, 'results:\t', errors,'mistakes, ', perc, '% correct')

    network.save_to_dat()

if __name__ == '__main__':
    std_exec()
