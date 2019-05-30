# Partial Scanning Transmission Electron Microscopy

This repository is for the [TODO](link goes here) "Partial Scan Electron Microscopy with Deep Learning". It contains source code for a multi-scale generative adversarial network that can complete 512x512 electron micrographs from partial scans. For partial scans with 1/20 coverage, it has a 2.6% intensity error.

<p align="center">
  <img src="adv_vs_non-adv.png">
</p>

## Architecture

<p align="center">
  <img src="simplified_gan.png">
</p>

Our training configuration can be partitioned into six subnetworks: an inner and outer generator, inner generator trainer and small, medium and large scale discriminators. The generators are all that is needed for inference.

## Example Usage

This short script is available as `inference.py` and gives an example of inference where the generator is loaded once to complete multiple scans:

```python
import numpy as np
from inference import Generator, get_example_scan, disp

#Use get_example_scan to select an example partial scan, ground truth pair from the project repository
#Try replacing this with your own partial scan, ground truth pair!
partial_scan, truth = get_example_scan() #Downloads example from GitHub - don't worry about moving the script

#Initialize generator so it's ready for repeated use
gen = Generator()

#Complete the scan
complete_scan = gen.infer(crop) 

#Generate can be reused multiple times once it has been initialised
# ... 

#Display results
disp(partial_scan) #Partial scan to be completed
disp(truth) #Ground truth
disp(complete_scan) #Scan completed by neural network
```

## Download

Training and inference scripts can be downloaded or cloned from this repository

```
git clone https://github.com/Jeffrey-Ede/partial-STEM.git
cd partial-STEM
```

The last saved checkpoint for a fully trained neural network is available [here](https://drive.google.com/open?id=1jkf9iSnarcuj2uRmsWmCEbghfncgWdXz).

## Dependencies

This neural network was trained using TensorFlow and requires it and other common python libraries. Most of these libraries come with modern python distributions by default. If you don't have some of these libraries, they can be installed using pip or another package manager. We used python version 3.6.

Libraries you need for both training and inference:

For training you also need:

* tensorFlow
* numpy
* cv2
* functools
* itertools
* collections
* six
* os
* argparse
* random
* scipy
* Image
* time
* PIL

## Training

To continue training the neural network; end-to-end or to fine-tune it, adjust some of the variables at the top of `train.py`. Specifically, variables indicating the location of your datasets and locations to save logs and checkpoints to. Note that there may be minor differences between the script and the paper due to active development. 

## Training Data

A training dataset with 161069 512x512 crops from STEM images is available upon request. Contact: {j.m.ede, r.beanland}@warwick.ac.uk. It will also be made available as a subset of the Warwick Large Electron Microscopy Dataset (WLEMD) in a future publication. A link will be provided here.
