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

take_ln = True
moving_avg = True
save = True
save_val = True
window_size = 2500
dataset_num = 8
mean_from_last = 20000
remove_repeats = True #Caused by starting from the same counter multiple times

scale = 1.0
ratio = 1.3 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()

#for i in range(1, 9):
#    log_loc = ("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-"+str(i)+"/")
#    notes_file = log_loc+"notes.txt"
#    with open(notes_file, "r") as f:
#        for l in f: print(l)

#7
#
#
#Reflection padding.
#10
#First convolution 3x3. ReLU activation.
#Better learning policy. Earlier beta decay.
#No non-linearity after final convolutions. Output in [-1,1].
#13
#KNN infilling. Note this does not make sense with noisy edges!
#Ancillary inner network trainer
#Signal combined with uniform noise to make low-duration areas less meaningful.
#16
#Spiral
#Wasserstein fine-tuning of non-spiral
#40x compression, lr 0.0002
#19
#40x compression, lr 0.00015
#Spiral, LR 0.00010, No Noise
#CReLU, 40x compression spiral
#22
#First kernel size decreased from 11 to 5
#3x3 kernels for first convolutions
#RMSProp
#25
#Momentum optimizer, momentum 0.9
#Momentum optimizer, momentum 0.0
#AdaGrad optimizer
#28
#Nesterov
#1/20 coverage spiral
#1/100 coverage spiral
#31
#1/10 coverage spiral
#1/40 coverage. Quartic loss with loss capping. Noisy.
#Quartic loss learning rate decreased from 0.00010 to 0.00005. Noisy.
#34
#40x compression with 1_000_000 training iterations. Noisy.
#1_000_000 iterations; learning rate decreased from 0.00010 to 0.00005. Noisy.
#1/20 coverage spiral. Adversarial training with 5x usual adversarial loss.
#37
#1/20 coverage spiral. Adversarial training with usual adversarial loss.
#Mean squared error without Huberization
#Weighted loss to try and reduce spatially systematic errors.
#40
#MSE without huberization, with capper
#No Huberization, with capper, adjusted for systematic errors
#Tiled input 10x to investigate overparameterization
#43
#Capped MSE. Repeat of 40
#Capped MSE. Losses adjusted based on systematic error. Repeat of 41
#
#46
#

labels_sets = [["Path Info, No Extra Residuals",
                "Path Info, Extra Residuals",
                "No Path Info, No Extra Residuals"],
               ["17x17 Kernels",
                "7x7 Kernels",
                "3x3 Kernels"],
               [r"75k-50k-175k, LR 0.0002, No Decay",
                r"200k-50k-150k, LR 0.0002, No Decay",
                r"270k-60k-120k, LR 0.0002, Decay",
                r"270k-60k-120k, LR 0.0004, Decay"],
               [r"Output in [$-$1,1]",
                r"Output in [0,1]"],
               [r"Later $\beta$ Decay",
                r"Earlier $\beta$ Decay"],
               [r"All convolutions 3x3, ReLU",
                r"No End Non-linearity",
                r"Unchanged (Reference)"],
               ["No Infilling",
                "KNN Infilling"],
               ["2 Stage with Fine Tuning",
                "1 Stage with Auxiliary Trainer"],
               ["Low Duration Unchanged",
                "Low Duration Added Noise"],
               ["Spiral, LR 0.00020",                
                "Spiral, LR 0.00015",
                "Grid, LR 0.00010",
                "Spiral, LR 0.00010",
                "Spiral, LR 0.00010, No Noise"],
               ["Momentum, 0.0",
                "Momentum, 0.9",
                "Nesterov, 0.9",
                "RMSProp", 
                "ADAM"],
               ["1/100 Spiral",
                "1/40 Spiral",
                "1/20 Spiral",
                "1/10 Spiral"],
               ["Quartic, Clipped, Train",
                "Quartic, Clipped, Val"],
               ["MSE",
                "MSE, Running",
                "Huber MSE",
                "Huber MSE, Clipped",
                "Huber MSE, Clipped, Running",
                "Huber MSE, Clipped, Final"]
              ]
sets = [[1, 2, 3], [5, 4, 23], [1, 4, 6, 7], 
        [7, 8], [7, 11], [10, 12, 11], 
        [12, 13], [12, 14], [14, 15], 
        [18, 19, 15, 16, 20], [26 ,25, 28, 24, 22], [30, 20, 29, 31],
        [33, 33], [38, 39, 36, 43, 44, 46] ]

losses_sets = []
iters_sets = []
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):

    if i != 13:
        continue

    print(labels)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=False)
    for j, dataset_num in enumerate(data_nums):
        log_loc = ("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-"+str(dataset_num)+"/")
        log_file = log_loc+"log.txt"
        val_file = log_loc+"val_log.txt"

        notes_file = log_loc+"notes.txt"
        with open(notes_file, "r") as f:
            for l in f: print(l)

        val_mode = labels[j] == "Quartic, Clipped, Val"

        switch = False
        losses = []
        vals = []
        losses_iters = []
        with open(log_file, "r") as f:

            if dataset_num < 14:
                for line in f:
                    numbers = re.findall(r"\[([-+]?\d*\.\d+|\d+)\]", line)
        
                    val_marks = re.findall(r"(\d+)NiN", line)
                    val_marks = [x is '1' for x in val_marks]

                    losses = [float(x) for x in numbers]
                    vals = [x for x, m in zip(losses, val_marks) if m]
                    losses = [x for x, m in zip(losses, val_marks) if not m]
                    losses_iters = [i for i in range(1, len(losses)+1)]
            else:
                numbers = []
                for line in f:
                    numbers += line.split(",")

                vals = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if "Val" in x]
                numbers = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if 'MSE:' in x]

                if val_mode:
                    losses = [min(float(x), 25.) for x, v in zip(numbers, vals) if int(v)]
                    losses_iters = [i for i, v in enumerate(vals) if int(v)]
                    print(losses_iters)
                else:
                    losses = [min(float(x), 25.) for x, v in zip(numbers, vals) if not int(v)]
                    losses_iters = [i for i in range(1, len(losses)+1)]
        try:
            switch = False
            val_losses = []
            val_iters = []
            with open(val_file, "r") as f:
                for line in f:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                    for i in range(1, len(numbers), 2):

                        val_losses.append(float(numbers[i]))
                        val_iters.append(float(numbers[i-1]))
        except:
            print("No val log {}".format(val_file))

        def moving_average(a, n):
            #a = np.concatenate([a[0]*np.ones((n)),a, a[0]*np.ones((n))])

            ret = np.cumsum(np.insert(a,0,0), dtype=float)
            return (ret[n:] - ret[:-n]) / float(n)

        if i in [13]:
            losses_iters = losses_iters[:500_000]
            losses = losses[:len(losses_iters)]
            
        avg_size = window_size if val_mode else window_size

        losses = moving_average(np.array(losses[:]), avg_size) if moving_avg else np.array(losses[:])
        losses_iters = moving_average(np.array(losses_iters[:]), avg_size) if moving_avg else np.array(losses[:])

        #val_losses = moving_average(np.array(val_losses[:]), avg_size) if moving_avg else np.array(val_losses[:])
        #val_iters = moving_average(np.array(val_iters[:]), avg_size) if moving_avg else np.array(val_iters[:])

        print(np.mean((losses[(len(losses)-mean_from_last):])[np.isfinite(losses[(len(losses)-mean_from_last):])]))

        #if take_ln:
        #    import math
        #    losses = math.log(losses)

        if val_mode:
            losses = losses[10000//50:]
            losses_iters = losses_iters[10000//50:]
        else:
            losses = losses[10000:]
            losses_iters = losses_iters[10000:]
        losses_iters = [i/1000 for i in losses_iters]
        losses_sets.append(losses)
        iters_sets.append(losses_iters)
        print(j, len(labels))

        if dataset_num == 14 and i in [7, 8]:
            losses_iters = losses_iters[:487_500]
            losses = losses[:len(losses_iters)]
        if dataset_num == 15 and i in [8]:
            losses_iters = losses_iters[:487_500]
            losses = losses[:len(losses_iters)]
        if dataset_num == 15 and i in [9]:
            losses_iters = losses_iters[:487_500]
            losses = losses[:len(losses_iters)]


        plt.plot(losses_iters, losses if take_ln else losses, label=labels[j], linewidth=3)
        #plt.plot(val_iters, np.log(val_losses) if take_ln else val_losses)

        if i in [5, 6, 7]:
            plt.ylim(3.7, 9.7)

        if i in [13]:
            plt.xlim(35, 65)
            plt.ylim(1.7, 3.3)

    plt.axvline(x=10, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Mean Squared Error-Based Loss')
    plt.xlabel('Training Iterations (x10$^3$)')
    if not i in [13]:
        plt.legend(loc='upper right', frameon=False, fontsize=8)

    #ax.tick_params('both')

    #plt.show()

    #f.set_size_inches(width, height)

    save_loc =  "Z:/Jeffrey-Ede/models/stem-random-walk-nin-figures/" + str(i+1) + ".png"
    plt.savefig( save_loc, bbox_inches='tight', )

    plt.gcf().clear()
