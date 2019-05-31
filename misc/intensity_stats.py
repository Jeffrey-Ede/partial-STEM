"""Analyse intensity distribution of original ARM crops."""

import numpy as np
from scipy.misc import imread

import os

import pickle

import matplotlib.pyplot as plt

PARENT = "F:/ARM_scans-crops/"
SUBSETS = ["val", "train", "test"]

STATS_SAVE_LOC = "H:/arm_scan_stats/"


def collect_data():

    for s in SUBSETS:
        dir = PARENT + s + "/"
        files = [dir+fn for fn in os.listdir(dir)]

        stats = {"max": [], "min": [], "stddev": [], "mean": []}
        for i, f in enumerate(files):
            print(f"{s} file {i} of {len(files)}")
            img = imread(f, mode='F')

            stats["max"].append(np.max(img))
            stats["min"].append(np.min(img))

            stats["stddev"].append(np.std(img))

            stats["mean"].append(np.mean(img))

        with open(STATS_SAVE_LOC+s+".P", "wb") as f:
            pickle.dump(stats, f)

def display_data():

    for s in SUBSETS:
        with open(STATS_SAVE_LOC+s+".P", "rb") as f:
            data = pickle.load(f)

        hist, edges = np.histogram(data["mean"], bins=128, normed=True)
        plt.plot(0.5*(edges[:len(hist)] + edges[1:len(hist)+1]), np.log10(hist+1.e-6))
    plt.show()

if __name__ == "__main__":

    #collect_data()
    display_data()