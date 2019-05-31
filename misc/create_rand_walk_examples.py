import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 9
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['axes.titlepad'] = 5
mpl.rcParams['savefig.dpi'] = 500

import matplotlib.mlab as mlab

import scipy.stats as stats

import cv2
from scipy.misc import imread

from PIL import Image, ImageDraw

import random

cropsize = 512
w = h = cropsize

cols = 4
rows = 2

max_cols = 3

set_num = 1
save_loc = "Z:/Jeffrey-Ede/models/stem-random-walk-nin-figures/rand_walk_examples.png"

# width as measured in inkscape
scale = 1.0
ratio = 1.3 # 1.618
width = scale * 2.2 * 3.487
height = 0.5 * width #0.6*4.35*(width / 1.618) / 2.2
num_data_to_use = 20000
num_hist_bins = 200

##Make a list of all the example images
loc = "Z:/Jeffrey-Ede/models/gan-infilling-100-STEM/"

#Put labels in order
col_labels = ["1/10", "1/20", "1/40", "1/100"]
labels = ["Spiral", "Gridlike"]

num_labels = len(labels)
print("Num labels: {}".format(len(labels)))

#Select subset of data
start_idx = (set_num-1)*rows*max_cols
end_idx = start_idx + cols

label_start_idx = (set_num-1)*rows

def gen_spiral(ratio, side=h, num_steps=10_000):
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
    k = 1/ (2*np.pi*ratio)

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

    #Blur path
    img = cv2.GaussianBlur(img,(3,3),0)

    return img

def gen_random_walk(channel_width, channel_height, amplitude=1, beta1=0., shift=0., steps=10):


    walk = np.zeros((int(np.ceil(channel_width+shift)), channel_height))
    halfway = (channel_width-1)/2
    center = halfway+shift
    size = int(np.ceil(channel_width+shift))

    mom = 0.
    y = 0.
    for i in range(channel_height):

        y1 = y
        #Get new position and adjust momentum
        step_y = random.randint(0, 1)
        if step_y == 1:
            mom = beta1*mom + (1-beta1)*amplitude*(1 + np.random.normal())
            y += mom
        else:
            y = amplitude*(-1 + np.random.normal())

        if y < -halfway:
            y = -halfway
            mom = -mom
        elif y > halfway:
            y = halfway
            mom = -mom

        #Move to position in steps
        y2 = y
        scale = np.sqrt(1+(y2-y1)**2)
        for j in range(steps):
            x = (j+1)/steps
            y = (y2-y1)*x + y1

            y_idx = center+y
            if y_idx != np.ceil(y_idx):
                if int(y_idx) < size:
                    walk[int(y_idx), i] += scale*(np.ceil(y_idx) - y_idx)/steps
                if int(y_idx)+1 < size:
                    walk[int(y_idx)+1, i] += scale*(1.-(np.ceil(y_idx) - y_idx))/steps
            else:
                walk[int(y_idx), i] = scale*1

    return walk, size

def gen_gridlike(use_frac, amp, extra=1.2):
    """Gridlike walk."""

    channel_height = int(extra*cropsize)

    steps = int(np.ceil(5*amp))

    channel_size = (2+np.sqrt(4-4*4*use_frac)) / (2*use_frac)
    num_channels = channel_height / channel_size
    mask = np.zeros( (channel_height, channel_height) )
    for i in range( int(num_channels) ):
        shift = i*channel_size - np.floor(i*channel_size)
        walk, size = gen_random_walk(
            channel_width=channel_size, 
            channel_height=channel_height, 
            amplitude=amp, 
            beta1=0.5, 
            shift=shift, 
            steps=steps)
        lower_idx = np.floor(i*channel_size)
        upper_idx = int(lower_idx) + size
        if upper_idx < channel_height:
            mask[int(lower_idx):upper_idx, :] = walk
        else:
            diff = int(upper_idx)-int(channel_height)
            mask[int(lower_idx):int(upper_idx)-diff, :] = walk[0:(size-diff), :]

    d = int(0.1*cropsize)
    mask = mask[d:(d+cropsize), d:(d+cropsize)]

    return mask.clip(0, 1)

#Generate random walks
data_to_use = []
for ratio in [1/10, 1/20, 1/40, 1/100]:
    walk = gen_spiral(ratio)
    data_to_use.append(walk)

for i, (use_frac, amp) in enumerate([(0.055, 1), (0.03, 1), (0.013, 1), (0.005, 1)]):
    walk = gen_gridlike(use_frac, amp)
    if i != 3:
        walk = walk + walk.T
    else:
        walk = walk + gen_gridlike(0.008, amp).T
    print("Walk Cover:", np.sum(walk)/(cropsize**2))
    data_to_use.append(walk)

#codes = [(num, 2, x+1) for x in range(2*num)]

f = plt.figure(1)
#f, big_axes = plt.subplots( figsize=(15.0, 15.0),nrows=1, ncols=1, sharey=True)

def plot_data(data, label, col_label, pos):

    ax = f.add_subplot(rows, cols, pos)

    img = data

    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    ax.set_frame_on(False)

    if pos % cols == 1:
        plt.ylabel(label, rotation=90)

    if pos <= 4:
        plt.title(col_label)

    return

#Plot the data
for i, data in enumerate(data_to_use, 1):
    idx = (i-1)//cols
    label = labels[idx]
    col_label = col_labels[(i-1)%cols]

    plot_data(data, label, col_label, i)

f.subplots_adjust(wspace=0.05, hspace=0.02)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

f.set_size_inches(width, height)

#plt.show()

f.savefig(save_loc, bbox_inches='tight', )

del f
