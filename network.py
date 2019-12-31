import numpy as np
import struct as st
from random import shuffle 
from load_mnist import load_all

import json
from matplotlib import pyplot as plt


class Brain:
    def __init__ (self, weight_Mat, biases):
        self.nLayers = len(biases)

        self.wMat = weight_Mat 
        self.bias = biases

            
    def read (self, v0):
        # result[k] = v[k] = W[k-1]*sigma(v[k-1])+b[k-1]
        # just the affine transformation, useful for gradients
        result = [v0] 
        # appo = sigma(result[k]) = sigma(v[k])
        appo = v0 
     
        for w,b in zip(self.wMat, self.bias):
            result.append(np.dot(appo, w.T) 
                          + np.tile(b, (v0.shape[0],1)))
            appo = sigma(result[-1])
        return result

    
    def guess (self,x):
        for w,b in zip(self.wMat,self.bias):
            x = sigma (w.dot(x) + b)
        return np.argmax(x)

        
    def SGD_training (self, 
                      samples, labels, 
                      eta = 0.5,
                      mu = 0.5,
                      batch_size = 10, 
                      regularize = 0, 
                      gamma = 0.5,
                     ):

        metrics = self.init_metrics()

        # splitting training data into batches
        indices = [k for k in range(batch_size, labels.shape[0], batch_size)]
        sample_batches = np.split(samples, indices)
        label_batches = np.split(labels, indices)

        # inintializing
        wSpeed = [np.zeros(np.shape(it)) for it in self.wMat]
        bSpeed = [np.zeros(len(it)) for it in self.bias]
        
        # training
        for batch, label_batch in zip(sample_batches, label_batches):
            # computing gradients
            result = self.read (batch)
            dW, db = self.gradErr(result, label_batch) 

            # adding to gradients' averages
            for j in range(self.nLayers):
                # computing new weights and biases with momentum
                wSpeed[j] = mu * wSpeed[j] - eta * np.mean(dW[j], axis=0)
                bSpeed[j] = mu * bSpeed[j] - eta * np.mean(db[j], axis=0)

                self.wMat[j] += wSpeed[j]
                self.bias[j] += bSpeed[j]

                # regularizing if requested
                if regularize == 1:
                    self.wMat[j] -= eta * gamma / len_data * np.sign(self.wMat[j])
                if regularize == 2:
                    self.wMat[j] *= 1 - eta * gamma / len_data 

            self.update_metrics(metrics, result, label_batch, dW, db)
        return metrics
      

    def gradErr (self, v, sol):
        dE = dC(sigma(v[-1]), sol) * dsigma(v[-1])
        Wgrad = [self.batch_tensordot(dE, sigma(v[-2]))]
        bgrad = [dE]

        for k in range (1, self.nLayers):
            dE = dE.dot(self.wMat[-k])
            dE *= dsigma(v[-k-1])
            Wgrad.append (self.batch_tensordot(dE,sigma(v[-k-2])))
            bgrad.append (dE)
        
        Wgrad.reverse()
        bgrad.reverse()
        return Wgrad, bgrad


    def batch_tensordot (self, x, y):
        result = np.empty((x.shape[0], x.shape[1], y.shape[1]))
        for k, (xk, yk) in enumerate(zip(x, y)):
            result[k] = np.tensordot(xk, yk, axes=0)
        return result


    def init_metrics (self):
        metrics = {
            'loss' : 0,
            'accuracy' : 0,
            'batch_loss' : [],
            'update_ratio' : []
        }
        return metrics


    def update_metrics (self, metrics, predict, lbl, gradW, gradb):
        metrics['batch_loss'].append(C(sigma(predict[-1]), lbl))
        metrics['loss'] += metrics['batch_loss'][-1]
        metrics['update_ratio'].append(
            sum([dw.sum() + db.sum() for dw, db in zip(gradW,gradb)]) 
            / sum([w.sum() + b.sum() for w,b in zip(self.wMat, self.bias)]))
        


    def save_to_json (self, file_name = '../saved_models/network3.json'):
        data = {"weights" : [w.tolist() for w in self.wMat],
                "biases" : [b.tolist() for b in self.bias]}
        
        outf = open (file_name, 'w') 
        json.dump(data, outf)
        outf.close()
        

#activation function and its derivative
#sigma = lambda x:  np.arctan(x) / np.pi + 0.5
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


def test (network, samples, labels):
    errors = 0
    for image, sol in zip(samples, labels):    
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


def plot_metrics (metrics, epoch_metrics, fig):
    metrics['loss'].append(epoch_metrics['loss'])
    metrics['batch_loss'] += epoch_metrics['batch_loss']
    metrics['update_ratio'] = epoch_metrics['update_ratio']

    epochs = np.arange(len(metrics['loss']))
    batches = np.arange(len(metrics['batch_loss']))
    epoch_batches = np.arange(len(metrics['update_ratio']))
   
    fig.clf()
    ax_loss = fig.add_subplot(223)
    ax_batch = fig.add_subplot(211)
    ax_update = fig.add_subplot(224)

    plt.subplots_adjust(hspace=0.3)

    ax_loss.plot(epochs, metrics['loss'])
    ax_batch.plot(batches, metrics['batch_loss'])
    ax_update.plot(epoch_batches, metrics['update_ratio'])

    ax_loss.title.set_text('Epoch Loss')
    ax_batch.title.set_text('Batch Loss')
    ax_update.title.set_text('Gradient / Weight ratio')

    plt.draw()
    plt.pause(0.001)


def init_weights (dims):
    weights, biases = [], []
    for i in range(len(dims)-1):
        weights.append(0.1 * np.random.randn(dims[i+1],dims[i]))
        biases.append(0.1 * np.random.randn(dims[i+1])) 
    return weights, biases


def std_exec ():
    # loading datasets
    images, labels, test_imgs, test_lbls = load_all()
    print('Finished loading data.')
    
    # neural network layers
    input_size = images[0].size
    dims = [input_size, 100, 10]
    weights, biases = init_weights(dims)

    network = Brain(weights, biases)

    # setting metrics 
    metrics = {'loss' : [], 'batch_loss' : [], 'update_ratio' : []}
    fig = plt.figure()
    plt.ion()
    plt.show()

    # training and testing
    reg = 0 
    for i in range(20):
        # shuffling training data
        training_data = np.concatenate([images, labels], axis=1) 
        shuffle (training_data)
        images = training_data[:, :input_size]
        labels = training_data[:, input_size:]

        epoch_metrics = network.SGD_training (images, labels, eta = 0.6)
        plot_metrics(metrics, epoch_metrics, fig)
        errors = test (network, test_imgs, test_lbls)

        perc = (1 - errors / len(test_lbls)) * 100
        print('Epoch {0} results: {1} mistakes, {2:.2f}% correct'
                .format(i + 1, errors, perc))

    plt.ioff()
    network.save_to_json('../saved_models/network3.json')


if __name__ == '__main__':
    std_exec()

#nice idea, though not feasible with dense nets, worth trying with convnets though
#if not i % 5 and i:
#    new_mat = np.eye(dims[1]) + 0.001 * np.random.randn(dims[1], dims[1])
#    new_bias = 0.001 * np.random.randn(dims[1])

#    network.nLayers += 1
#    network.wMat.insert(-1, new_mat)
#    network.bias.insert(-1, new_bias)
