import numpy as np
from scipy.misc import imread
from scipy.stats import entropy

import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['savefig.dpi'] = 200
fontsize = 10
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import cv2

from PIL import Image
from PIL import ImageDraw

from scipy.ndimage.filters import laplace

image_locs = [
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-43/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-39/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-44/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-46/errors.tif",
    "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-36/errors.tif"]
x_titles = [
    "Clipped",
    "Running",  
    "Clipped, Running",
    "Clipped, Fixed",
    "Adversarial"]

def image_entropy(img, num_bins=2**8, eps=1.e-8):

    hist, _ = np.histogram(img.astype(np.float32), bins=1024)
    entr = entropy(hist + eps, base=2) #Base 2 is Shannon entropy

    return entr

images = [imread(loc, mode="F") for loc in image_locs]
imgs = []
for img in images:

    norm_img = 0.2*(img - np.mean(img))/np.std(img)
    norm_img += 0.5

    var_lap = np.std(laplace(norm_img))**2

    print("mean, std, var_lap:", np.mean(img), np.std(img), var_lap)

    imgs.append(norm_img)

    norm_img = 256*np.random.rand(512,512)

    dx = cv2.Sobel(norm_img, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(norm_img, cv2.CV_64F, 0, 1)

    max_grad = max(np.max(np.abs(dx)), np.max(np.abs(dy))) + 1; print(max_grad)
    num_bins = 1024

    dx = 0.5*(dx+max_grad)
    dy = 0.5*(dy+max_grad)

    grads = np.zeros((num_bins,num_bins))

    for i in range(512):
        for j in range(512):
            x = dx[i,j]
            y = dy[i,j]
            grads[int(num_bins*x/max_grad),int(num_bins*y/max_grad)] += 1
    grads /= num_bins**2


    print(0.5*image_entropy(grads))

    #Local Shannon entropy
    #step = 16
    #entr = 0
    #num_ents = 0
    #for x in range(0,512,step):
    #    for y in range(0,512,step):
    #        num_ents += 1
    #        entr += image_entropy(norm_img[x:x+step, y:y+step])
    #entr /= num_ents

    #print(image_entropy(norm_img))

def inspiral(coverage, side, num_steps=10_000):
    """Duration spent at each location as a particle falls in a magnetic 
    field. Trajectory chosen so that the duration density is (approx.)
    evenly distributed. Trajectory is calculated stepwise.
    
    Args: 
        coverage: Average amount of time spent at a random pixel
        side: Sidelength of square image that the motion is 
        inscribed on.

    Returns:
        Amounts of time spent at each pixel on a square image as a charged
        particle inspirals.
    """
    
    #Use size that is larger than the image
    size = int(np.ceil(np.sqrt(2)*side))

    #Maximum radius of motion
    R = size/2

    #Get constant in equation of motion 
    k = 1/ (2*np.pi*coverage)

    #Maximum theta that is in the image
    theta_max = R / k

    #Equispaced steps
    theta = np.arange(0, theta_max, theta_max/num_steps)
    r = k * theta

    #Convert to cartesian, with (0,0) at the center of the image
    x = r*np.cos(theta) + R
    y = r*np.sin(theta) + R

    #Draw spiral
    z = np.empty((x.size + y.size,), dtype=x.dtype)
    z[0::2] = x
    z[1::2] = y

    z = list(z)

    img = Image.new('F', (size,size), "black")
    img_draw = ImageDraw.Draw(img)
    img_draw = img_draw.line(z)
    
    img = np.asarray(img)
    img = img[size//2-side//2:size//2+side//2+side%2, 
              size//2-side//2:size//2+side//2+side%2]

    return img


def scale0to1(img):
    
    min = np.min(img)
    max = np.max(img)

    print(min, max)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def block_resize(img, new_size):

    x = np.zeros(new_size)
    
    dx = int(new_size[0]/img.shape[0])
    dy = int(new_size[1]/img.shape[1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i,j]

            for u in range(dx):
                for v in range(dy):
                    x[i*dx+u, j*dy+v] = px

    return x
    
#Width as measured in inkscape
scale = 4
width = scale * 2.2
height = scale* (width / 1.618) / 2.2

print(np.mean(imgs[0]), np.mean(imgs[1]))

#Image.fromarray(imgs[1]).save('general_abs_err.tif')

set_mins = []
set_maxs = []

for img in imgs:
    set_mins.append(0)
    set_maxs.append(1)

w = h = 512

subplot_cropsize = 64
subplot_prop_of_size = 0.6
subplot_side = int(subplot_prop_of_size*w)
subplot_prop_outside = 0.2
out_len = int(subplot_prop_outside*subplot_side)
side = w+out_len

print(imgs[1])

f=plt.figure(figsize=(1, 4))
columns = 5
rows = 1

#spiral = inspiral(1/20, int(512*0.6*512/64))
#spiral_crop = spiral[:subplot_side, :subplot_side]

for i in range(1):
    for j in range(1, columns+1):
        img = np.ones(shape=(side,side))
        img[:w, :w] = scale0to1(imgs[columns*i+j-1])
        crop = cv2.resize(img[:subplot_cropsize, :subplot_cropsize], 
                          (subplot_side, subplot_side),
                          cv2.INTER_CUBIC)
        img[(side-subplot_side):,(side-subplot_side):] = crop
        img = img.clip(0., 1.)
        k = i*columns+j
        ax = f.add_subplot(rows, columns, k)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])

        ax.set_frame_on(False)
        if not i:
            ax.set_title(x_titles[j-1])#, fontsize=fontsize)

f.subplots_adjust(wspace=0.035, hspace=0.0)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)
#f.tight_layout()

f.set_size_inches(width, height)

#plt.show()

f.savefig('systematic_errors.pdf', bbox_inches='tight')
