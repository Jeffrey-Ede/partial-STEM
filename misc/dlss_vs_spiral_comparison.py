import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 9
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = [4,3]

import scipy.stats as stats

from functools import reduce
from math import gcd
a = [16, 25, 36, 64, 100]   #will work for an int array of any length
from math import gcd

def getLCM(a, b):
    return a * b // gcd(a, b)

from functools import reduce

def nlcm(nums):
    return reduce(getLCM, nums, 1)
print(nlcm(a))

data_num = 50

scale = 1.0
ratio = 1.618 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()
ax = f.add_subplot(111)

data = np.load(f"Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-{data_num}/mses.npy")
data = data[:20_000]
data = 100*np.sqrt(data/200) / 2

values0 = [0.0313089676, 0.0334226, 0.0381019338965, 0.0404207669, 0.0433954969]
#labels0 = [1/16, 1/25, 1/36, 1/64, 1/100]
labels0 = [1/16, 1/24.70958620039589, 1/35.44402379664684, 1/64, 1/96.94674556213018]

values1 = [0.03768744692206383, 0.0411827526986599, 0.04505755379796028, 0.04857335239648819, 
           0.05248530954122543, 0.05585743114352226, 0.06210533529520035]
stds1 = [0.09850746393203735, 0.09967660158872604, 0.10715330392122269, 0.1089765802025795, 
         0.11647973209619522, 0.11598793417215347, 0.11943922191858292]
labels1 = [1/16, 1/25, 1/36, 1/49, 1/64, 1/81, 1/100]
labels1 = [17.859653903801608, 27.264066562662506, 38.23012979437072, 49.98932112890923, 
           60.51338873499538, 73.65664512503513, 87.0332005312085]
labels1 = [1/x for x in labels1]

print(labels0)
plt.plot(labels1, values1, 'x', label="Spiral")
plt.plot(labels0, values0, 'o', label="DLSS")

plt.xlabel('Coverage')
plt.ylabel('Root Mean Squared Error')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

#plt.xlim(0, 0.072)
#plt.ylim(0.0304, 0.045)

plt.tick_params()

plt.legend(loc='upper right', frameon=False, fontsize=8, bbox_to_anchor=(0.96, 0.95))

#plt.show()

#for code, data in zip(codes, datasets):
#    subplot_creator(code, data)

#f.subplots_adjust(wspace=0.18, hspace=0.26)
#f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

#ax.set_ylabel('Some Metric (in unit)')
#ax.set_xlabel('Something (in unit)')
#ax.set_xlim(0, 3*np.pi)

#f.set_size_inches(width, height)

#plt.show()
#quit()
save_loc = 'dlss_vs_spiral_comparison.png'
plt.savefig( save_loc, bbox_inches='tight', )

