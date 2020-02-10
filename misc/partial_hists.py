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
#mpl.rcParams['title.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = [4, 3]

take_ln = True
moving_avg = True
save = True
save_val = True
window_size = 2500
dataset_num = 8
mean_from_last = 20000
remove_repeats = True #Caused by starting from the same counter multiple times

scale = 1
ratio = 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()

labels_sets = [["1/17.9", "1/27.3", "1/38.2", "1/50.0", "1/60.5", "1/73.7", "1/87.0"]]


sets = [[74, 71, 75, 72, 77, 73, 76]]

f = plt.figure()
ax = f.add_subplot(111)

losses_sets = []
iters_sets = []
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):

    #ax._frameon = False

    #ax = f.add_subplot(1, 2, i+1)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=False)
    ax.minorticks_on()
    for j, dataset_num in enumerate(data_nums):
        if not i:
            hist_loc = ("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-"+str(dataset_num)+"/")
            hist_file = hist_loc+"mses.npy"
        else:
            hist_file = f"Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-68/mses-{dataset_num}.npy"

        mses = np.load(hist_file) /2
        #for x in mses: print(x)
        #print(np.mean(mses), np.std(mses))
        #print(len([x for x in mses if x >= 10]))
        mses = [np.sqrt(x) for x in mses if x < 0.05]

        bins, edges = np.histogram(mses, 100)
            
        edges = 0.5*(edges[:-1] + edges[1:])

        ax.plot(edges, bins, label=labels[j], linewidth=1)

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Root Mean Squared Error')

    #ax.set_ylim(-100, 2175)

    plt.legend(loc='upper right', frameon=False, fontsize=8)

    plt.minorticks_on()
    
#f.subplots_adjust(wspace=0.22, hspace=0.26)
#f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

#f.set_size_inches(width, height)

save_loc =  "Z:/Jeffrey-Ede/models/stem-random-walk-nin-figures/partial_hist.png"
plt.savefig( save_loc, bbox_inches='tight', )

#plt.gcf().clear()
