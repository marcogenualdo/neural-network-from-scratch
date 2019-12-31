import struct as st
import numpy as np


data_path = '/home/jonny/Documents/pyfiles/nns/mnist/data/'
training_filenames = (data_path + 'train-images-idx3-ubyte', data_path + 'train-labels-idx1-ubyte') 
test_filenames = (data_path + 't10k-images-idx3-ubyte', data_path + 't10k-labels-idx1-ubyte')


def one_hot (numbers, dim):
    vectors = np.zeros((numbers.shape[0], dim))

    for v,n in zip(vectors, numbers):
        v[n] = 1;
    return vectors


def load_images (img_name, 
                 lbl_name, 
                 vectorize_labels = 0,
                 load_up_to = 'all',
                ):

    img_file = open(img_name, 'rb')
    lbl_file = open(lbl_name, 'rb')

    img_file.seek(0) #getting to the beginning of the file
    lbl_file.seek(8) #labels start from the 8-th byte of this file
    magic_n = st.unpack('>4B', img_file.read(4)) #magic number

    nImgs = st.unpack('>I',img_file.read(4))[0] #number of images
    nRows = st.unpack('>I',img_file.read(4))[0] #number of rows
    nCols = st.unpack('>I',img_file.read(4))[0] #number of columns
    if load_up_to != 'all': nImgs = int(load_up_to)

    #array of images
    images = np.empty((nImgs,nRows*nCols))
    images =  np.asarray(st.unpack('>'+'B'*nRows*nCols*nImgs,
    img_file.read(nImgs*nRows*nCols))).reshape((nImgs,nRows*nCols)) / 255
    
    labels = np.asarray(st.unpack('>'+str(nImgs) + 'B',lbl_file.read(nImgs))).reshape(nImgs)

    #array of labels
    if vectorize_labels: labels = one_hot(labels, vectorize_labels)

    img_file.close()
    lbl_file.close()
    return images, labels


def load_all (load_up_to = 'all'):
    imgs, lbls = load_images(
        *training_filenames,
        vectorize_labels=10, 
        load_up_to=load_up_to,
    )

    test_imgs, test_lbls = load_images(
        *test_filenames, 
        load_up_to=load_up_to, 
    )

    return imgs, lbls, test_imgs, test_lbls 


def display (images, time):
    from matplotlib import pyplot as plt
    
    plt.ion()
    plt.show()
    for img in images:
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.draw()
        plt.pause(time)
    plt.ioff()


def _test_these_functions():
    imgs, lbls, timgs, tlbls = load_all(load_up_to=10)
    display(imgs, 0.8)
