### DeepSol: A Deep Learning Framework for Sequence-Based Protein Solubility Prediction

![alt text](http://people.csail.mit.edu/sameerk/dsol.svg)

## Motivation
Protein solubility can be a decisive factor in both research and production efficiency. Novel in silico, accurate, sequence-based protein solubility predictors are highly sought.

## Installation

### Requirements

This step will install all the dependencies required for running DeepSol in an Anaconda virtual environment locally. You do not need sudo permissions for this step.

  - Install Anaconda
    1. Download Anaconda (64 bit) installer python3.x for linux : https://www.anaconda.com/download/#linux
    2. Run the installer : `bash Anaconda3-5.0.1-Linux-x86_64.sh` and follow the instructions to install anaconda at your preferred location
    3. Run `conda update conda`

  - Creating the environment 
    1. Run `git clone https://github.com/sameerkhurana10/DSOL_rv0.2.git`
    2. Run `cd DSOL_rv0.2`
    3. Run `export PATH=<your_anaconda_folder>/bin:$PATH`
    4. Run `conda env create -f environment.yml`
    5. Run `source activate dsol`
  
  - SCRATCH (version SCRATCH-1D release 1.1) (http://scratch.proteomics.ics.uci.edu, Downloads: http://download.igb.uci.edu/#sspro)


All operations related to DeepSol models are to be performed from the folder `DSOL_rv0.2`.

### Recipe for running DeepSol

Recipe is contained in the script `run.sh`. To see the options run `./run.sh` and you shall see the following:

```
main options (for others, see top of script file)
  --model (deepsol1/deepsol2/deepsol3)               # model architecture to use
  --mode  (preprocess/train/decode/cv)                     # data preparation or decode or cross-validate using an existing model
  --stage (1/2)                                      # point to run the script from 
  --conf_file                                        # model parameter file
  --keras_backend                                    # backend for keras
  --cuda_root                                        # the path cuda installation
  --device (cuda/cpu)                                # device to use for running the recipe
```
There are two stages in the script. 

1. Data preparation
2. Model building with `--mode train` and decoding with best DeepSol models using `--mode decode`. Information about `--mode cv` is given in "parameter variance check" section.

We provide support for gpu usage using the option `--device cuda`. More details in the GPU section.


## CPU

### Train New Models

Train DeepSol models using pre-compiled training, validation data and optimal hyper-parameter setting as in `parameters.json` file:

  1. `./run.sh --model deepsol1 --stage 2 --mode train --device cpu data/protein.data` 
  2. `./run.sh --model deepsol2 --stage 2 --mode train --device cpu data/protein_with_bio.data`

Result will be a model named `deepsol1` or `deepsol2` stored in `results/models/`. 

Note that we used `--model deepsol2`, you can use `deepsol3` for 2. Ignore `UserWarning` at the output.

### Test Best DeepSol Models

Test existing DeepSol models with pre-compiled test data:

  1. `./run.sh --model deepsol1 --stage 2 --mode decode --device cpu data/protein.data`
  2. `./run.sh --model deepsol2 --stage 2 --mode decode --device cpu data/protein_with_bio.data` 

Result will be saved in `results/reports/`.

Note that we used `--model deepsol2`, you can use `deepsol3` for 2. 


## GPU

### Cuda installation

Ensure that cuda is installed. We support Cuda 8.0 and Cudnn 5.1 . For any other version of Cudnn, you might run into some issues.

Install Cuda 8.0 and Cudnn 5.1 from https://developer.nvidia.com/

Code was tested against GeForce GTX 1080 Nvidia GPUs https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080/ . The GPU driver version is 384.59.

Code was also tested on Nvidia Tesla K20Xm : https://www.techpowerup.com/gpudb/1884/tesla-k20xm with driver version 375.66.

### Train New Models

Train DeepSol models using pre-compiled training, validation data and optimal hyper-parameter setting as in `parameters.json` file:

  1. `./run.sh --model deepsol1 --stage 2 --mode train --cuda_root <path-to-your-cuda-installation> --device cuda data/protein.data`
  2. `./run.sh --model deepsol2 --stage 2 --mode train --cuda_root <path-to-your-cuda-installation> --device cuda data/protein_with_bio.data`.

Result will be a model named `deepsol1` or `deepsol2` stored in `results/models`.

Note that we used `--model deepsol2`, you can use `deepsol3` for 2. Ignore `UserWarning` at the output.

Also, `--cuda_root` should be the path to your cuda installation. By default it is `/usr/local/cuda`.

### Test Best DeepSol Models

Test existing DeepSol models with pre-compiled test data:

  1. `./run.sh --model deepsol1 --stage 2 --mode decode --cuda_root <path-to-your-cuda-installation> --device cuda data/protein.data`
  2. `./run.sh --model deepsol2 --stage 2 --mode decode --cuda_root <path-to-your-cuda-installation> --device cuda data/protein_with_bio.data`.

Result will be saved in `results/reports/`.

Note that we used `--model deepsol2`, you can use `deepsol3` for 2. 


## Parameter Variance Check

In this section we calculate the variance in performance of the DeepSol models on 10 cross-validation folds for dataset used in our paper.

For CPU: 

  1. run `./run.sh --model deepsol1 --stage 2 --mode cv --device cpu data/protein.data`
  2. run `./run.sh --model deepsol2 --stage 2 --mode cv --device cpu data/protein_with_bio.data`

For GPU:

  1. run `./run.sh --model deepsol1 --stage 2 --mode cv --cuda_root <path-to-your-cuda-installation> --device cuda data/protein.data` 
  2. run `./run.sh --model deepsol2 --stage 2 --mode cv --cuda_root <path-to-your-cuda-installation> --device cuda data/protein_with_bio.data`

Result will be saved in `results/reports/`.

Note that we used `--model deepsol2`, you can use `deepsol3` for 2. 
