from numpy import reshape
from pathlib import Path
from minisom import MiniSom
import matplotlib.pyplot as plt
import glob

# read the train image
img = plt.imread('images/train_lv_2013.png')

#normalize the image data, as a pre processing step, where necessary.
def normalize(image):
    image_rng = image.max()-image.min()
    image_min = image.min()
    return (image-image_min)*255/image_rng
#img_norm = normalize(img) - for this example, normalization is not  required

#Then reshape the image matrix - MiniSom implementation requires the input data to be 
#a Numpy matrix where each row corresponds to a record or as list of lists

pix = reshape(img,(img.shape[0]*img.shape[1],3))

#SOM initialization and training
print('initializing a 4 by 4 SOM...')
som = MiniSom(4,4,3,sigma=1.2,learning_rate=0.2) # 4x4 = 16 models
som.random_weights_init(pix)
print('Training the SOM with 1000 iterations...')
som.train_batch(pix,1000)
print ('... training completed')

#Get the image names from their folder
test_images = glob.glob("./images/mead/*.png")

#Determine SOM-QE values
print('Determining the SOM-QE ..')
Qerror = [] #to keep the list of SOM-QE values
image_name = [] #store image names
for im in sorted(test_images):
    image = plt.imread(im)
    im_path = Path(im)
    im_name = im_path.stem
    pixa = reshape(image,(image.shape[0]*image.shape[1],3))
    qe=som.quantization_error(pixa)
    Qerror.append(qe)
    image_name.append(im_name)
print('... SOM-QE values calculated')

#plot the results: Year against SOM-QE value
print('Plotting the results ..')    
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)
ax.plot(image_name,Qerror)
plt.title('SOM-QE determination of water level trends in Lake Mead')
plt.xlabel('Year image was taken')
plt.ylabel('SOM-QE value of the image')
plt.savefig('results/som-qe_mead.png')
plt.show()
print('... done')
