from pylab import imread,imshow,figure,show,subplots,title,setp,tick_params,axis,NullFormatter,NullLocator
from numpy import reshape,flipud,unravel_index,zeros,savetxt,asarray,column_stack,genfromtxt
import time

from minisom import MiniSom
start_time = time.time()

# read the train image
img = imread('train_lv_2013.png')

#normalize the image data, as a pre processing step, where necessary.
def normalize(image):
    image_rng = image.max()-image.min()
    image_min = image.min()
    return (image-image_min)*255/image_rng
#img_norm = normalize(img) - for this example, normalization is not  required

#Then reshape the pix matrix - MiniSom implementation requires the input data to be 
#a Numpy matrix where each row corresponds to a record or as list of lists

pix = reshape(img,(img.shape[0]*img.shape[1],3))

# SOM initialization and training
print('initializing a 4 by 4 SOM...')
som = MiniSom(4,4,3,sigma=1.2,learning_rate=0.2) # 4x4 = 16 models
som.random_weights_init(pix)
print('Training the SOM with 1000 iterations...')
som.train_batch(pix,1000)
print ('done')
print('Determining the SOM-QE value...')
qe=som.quantization_error(pix)
print('The SOM-QE value of the image is: ', qe)
#for a series of images, this can be stored in a file.
