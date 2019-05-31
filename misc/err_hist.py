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

scale = 1.0
ratio = 1.3 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()
ax = f.add_subplot(111)

data = np.load("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-43/mses.npy")
data = data[:20_000]
data = 100*np.sqrt(data/200) / 2
print(np.mean(data), np.std(data), np.sum(data > 2.4912944+5*0.873168)/20_000)
print(data.shape)
plt.hist(data, 50, normed=None, facecolor='grey', edgecolor='black', alpha=0.75, linewidth=1)
#plt.show()

plt.xlabel('RMS Intensity Error (%)')
plt.ylabel('Frequency')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
plt.tick_params()

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
save_loc = '20x_err_hist.png'
plt.savefig( save_loc, bbox_inches='tight', )

