from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn

import numpy as np



"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function=None, random_seed=3000):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            decay_function, function that reduces learning_rate and sigma at each iteration
                            default function: lambda x,current_iteration,max_iter: x/(1+current_iteration/max_iter)
            random_seed, random seed to use.
        """
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self.random_generator = random.RandomState(random_seed)
        else:
            self.random_generator = random.RandomState(random_seed)
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = self.random_generator.rand(x,y,input_len)*2-1 # random initialization
        for i in range(x):
            for j in range(y):
                self.weights[i,j] = self.weights[i,j] / fast_norm(self.weights[i,j]) # normalization
        self.activation_map = zeros((x,y))
        self.neigx = arange(x)
        self.neigy = arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = subtract(x, self.weights) # x - w
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  # || x - w ||
            it.iternext()

    def activate(self, x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def gaussian(self, c, sigma):
        """ Returns a Gaussian centered in c """
        d = 2*pi*sigma*sigma
        ax = exp(-power(self.neigx-c[0], 2)/d)
        ay = exp(-power(self.neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def diff_gaussian(self, c, sigma):
        """ Mexican hat centered in c (unused) """
        xx, yy = meshgrid(self.neigx, self.neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def winner(self, x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    #p = weights[it.multi_index]
    #g = self.neighborhood(win, sig)*eta # improves the performances
    #it = nditer(g, flags=['multi_index'])
    
   
    def update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        eta = self._decay_function(self.learning_rate, t, self.T)
        sig = self._decay_function(self.sigma, t, self.T) # sigma and learning rate decrease with the same rule
        g = self.neighborhood(win, sig)*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

       
    def quantization(self, data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[self.random_generator.randint(len(data))]
            self.weights[it.multi_index] = self.weights[it.multi_index]/fast_norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_iteration):
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            rand_i = self.random_generator.randint(len(data)) # pick a random sample
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data, num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2  # keeps the learning rate nearly constant for the last half of the iterations

    def distance_map(self):
        """ Returns the distance map of the weights.
            Each cell is the normalised sum of the distances between a neuron and its neighbours.
        """
        co_x=[]
        co_y=[]
        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += fast_norm(self.weights[ii, jj, :]-self.weights[it.multi_index])
                    co_y.append(jj)
                co_x.append(ii)
            it.iternext()
            
        um = um/um.max()
        return um,co_x,co_y

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error_compare(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        err_ind = []
        pix = []
        for x in data:
            err = fast_norm(x-self.weights[self.winner(x)])#euclidean dist = numpy.linalg.norm(a-b)
            err_ind.append(err)
            pix.append(x)
            #pix.update({err:x})
            #error += err
        #return error/len(data), err_ind
        return err_ind,pix
    def quantization_error(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        err_ind = []
        for x in data:
            err = fast_norm(x-self.weights[self.winner(x)])#euclidean dist = numpy.linalg.norm(a-b)
            err_ind.append(err)
            error += err
        return error/len(data),len(data)
        

    def win_map(self, data):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

from pylab import imread,imshow,figure,show,subplots,title,setp,tick_params,axis,NullFormatter,NullLocator
from numpy import reshape,flipud,unravel_index,zeros,savetxt,asarray,column_stack,genfromtxt
import time

start_time = time.time()

# read the train image
img = imread('train_lv_2013.png')
#img = imread('check_images/1/SampleReferenceImage.png')

#normalize the image data
def normalize(image):
    image_rng = image.max()-image.min()
    image_min = image.min()
    return (image-image_min)*255/image_rng

# ... and reshaping the pixels matrix
#img_norm = normalize(img)
pixels = reshape(img,(img.shape[0]*img.shape[1],3))

#pixels = normalize(img)

# SOM initialization and training
print('training...')
som = MiniSom(4,4,3,sigma=1.2,learning_rate=0.2) # 4x4 = 16 models
som.random_weights_init(pixels)
som.train_batch(pixels,10000)
print ('done')
print

#...............Cluster Analysis begins here ..........
print ('Cluster analysis ... begins ')
print('get the win_map...')
Qerror = []
keying = []
dic = som.win_map(pixels) # get list of all patterns at particular position 
print('get the QE for each position...')

count = 0
for key,value in sorted(dic.items()):
    qe=som.quantization_error(value)
    count = count + 1
    
    print ('Cluster number: ', count)
    print ('QE and size of cluster is: ', qe)#(er_value/len(value)),len(value))
    
    Qerror.append(qe)
    
    keying.append(key)
print (len(dic), 'clusters')
print()

dataQE=[item[0] for item in Qerror] #separate qe from size
dataSize=[item[1] for item in Qerror]

da = column_stack((keying,Qerror))
savetxt('2003.csv',da,fmt='%s')

keyQE = column_stack((keying,dataQE))
savetxt('2004.csv',keyQE,fmt='%s')

k = np.append(da, keyQE, axis=1)

import pandas as pd
import glob

train = pd.DataFrame(da)
pd.set_option('display.max_columns', 300)
test_images = glob.glob("/home/boss18/Documents/som/jobs/test/*.png")
for im in sorted(test_images):
    
    image = imread(im)
    pix = reshape(image,(image.shape[0]*image.shape[1],3))
    clusters = som.win_map(pix) # get list of all patterns at particular position
    count = 0
    QE=[]
    
    for key,value in sorted(clusters.items()):
        qe=som.quantization_error(value)
        count = count + 1
    
        
        print ('In test case, QE and size of cluster is: ', qe)#(er_value/len(value)),len(value))

        QE.append(qe)

    dataQE=[item[0] for item in QE]  
    train[im] = pd.Series(dataQE, index=train.index)
    print(count)
    print(train)
    

print ('End .... cluster analysis')

