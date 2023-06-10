from pylab import imread,imshow,figure,show,subplots,title,setp,tick_params,axis,NullFormatter,NullLocator
from numpy import reshape,flipud,unravel_index,zeros,savetxt,asarray,column_stack,genfromtxt
import time,os,glob
from pathlib import Path

from minisom import MiniSom
import matplotlib.pyplot as plt
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
test_images = glob.glob("./mead/*.png")
Qerror = []
image_name = []
for im in sorted(test_images):
    print(im)
    image = imread(im)
   
    ii=Path(im)
    i = ii.stem
    print(i,ii)
    pixa = reshape(image,(image.shape[0]*image.shape[1],3))
    qe=som.quantization_error(pixa)
    print(qe)
    Qerror.append(qe)
    image_name.append(i)
print(Qerror)
print(image_name)
#plot the results: Year against SOM-QE value
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)
ax.plot(image_name,Qerror)
plt.title('SOM-QE determination of water level trends in Lake Mead')
plt.xlabel('Year image was taken')
plt.ylabel('SOM-QE value of the image')
plt.show()
