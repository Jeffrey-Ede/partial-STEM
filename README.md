# Partial Scanning Transmission Electron Microscopy

This repository is for the [paper](https://arxiv.org/abs/1807.11234) "Partial Scan Electron Microscopy with Deep Learning". It contains source code for a multi-scale generative adversarial network that can complete 512x512 electron micrographs from partial scans. For partial scans with 1/20 coverage, it has a 2.6% intensity error. Inference takes 20-30 ms, enabling real-time partial scan completion.

<p align="center">
  <img src="adv_vs_non-adv.png">
</p>

## Architecture

<p align="center">
  <img src="simplified_gan.png">
</p>

Our training configuration can be partitioned into six subnetworks: an inner and outer generator, inner generator trainer and small, medium and large scale discriminators. The generators are all that is needed for inference.
