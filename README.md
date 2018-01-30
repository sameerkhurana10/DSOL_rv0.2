### DeepSol: A Deep Learning Framework for Sequence-Based Protein Solubility Prediction

![alt text](https://zenodo.org/badge/DOI/10.5281/zenodo.1162655.svg)

![alt text](http://people.csail.mit.edu/sameerk/dsol.svg)

## Motivation
Protein solubility can be a decisive factor in both research and production efficiency. Novel in silico, accurate, sequence-based protein solubility predictors are highly sought.

# Installation

### Requirements

This step will install all the dependencies required for running DeepSol in an Anaconda virtual environment locally. You do not need sudo permissions for this step.

  - Install Anaconda
    1. Download Anaconda (64 bit) installer python3.x for linux : https://www.anaconda.com/download/#linux
    2. Run the installer : `bash Anaconda3-5.0.1-Linux-x86_64.sh` and follow the instructions to install anaconda at your preferred location 
    3. Need conda > 4.3.30 (If conda already present but lower than this version do: conda upgrade conda)

  - Creating the environment 
    1. Run `git clone https://github.com/sameerkhurana10/DSOL_rv0.2.git`
    2. Run `cd DSOL_rv0.2`
    3. Run `export PATH=<your_anaconda_folder>/bin:$PATH`
    4. Run `conda env create -f environment.yml`
    5. Run `source activate dsol`

  - R requirements
    - Run R REPL by running the following: `R`
    -  Install R libraries
       1.  Interpol (do `install.packages('Interpol')` )
       2.  bio3d    (do `install.packages('bio3d')` )
       3.  doMC     (do `install.packages('doMC')`)
       
    Quit R REPL: `quit()` 
  
  - SCRATCH (version SCRATCH-1D release 1.1) (http://scratch.proteomics.ics.uci.edu, Downloads: http://download.igb.uci.edu/#sspro)
    1. Run `wget http://download.igb.uci.edu/SCRATCH-1D_1.1.tar.gz`
    2. Run `tar -xvzf SCRATCH-1D_1.1.tar.gz`
    3. Run `cd SCRATCH-1D_1.1`
    4. Run `perl install.pl`
    5. Run `cd ..`

All operations related to DeepSol models are to be performed from the folder `DSOL_rv0.2`

# Run DeepSol on New Test file

To run DeepSol on your own protein sequences you need the following two things:

  1. Protein Sequence File: Protein sequence/sequences of interest in fasta format (https://en.wikipedia.org/wiki/FASTA_format). We provide `data/Seq_solo.fasta` as an example 
  2. SCRATCH: Software used to extract biological features from a given protein sequence file. Follow instructions in the previous section to Install SCRATCH

### Execute in the command line
 
  1. `R --vanilla < scripts/PaRSnIP.R data/Seq_solo.fasta <path-to-your-scratch-installation>/bin/run_SCRATCH-1D_predictors.sh new_test 32`
  
  `32` is the number of processors, `new_test` is the output files' prefix

Following this step, two files are created in the `data` folder: 
* `new_test_src` : contains raw protein sequences
* `new_test_src_bio` : contains biological features corresponding to the raw protein sequences

**Note**: `data/Seq_multi.fasta` can be used instead of `data/Seq_solo.fasta`. `Seq_multi.fasta` has multiple protein sequences
  
  2. `./run.sh --model deepsol1 --stage 1 --mode preprocess --device cpu --test_file new_test data/newtest.data`

This step Preprocesses data files from step 1., and stores at `data/newtest.data` in a format acceptable to Deepsol models. 

**Note**: You can also use `deepsol2` or `deepsol3` in place of `deepsol1`. See Paper for more details

  3. `./run.sh --model deepsol1 --stage 2 --mode decode --device cpu data/newtest.data`

Result will be saved in `results/reports/`. Note: you can also use `deepsol2` or `deepsol3` in place of `deepsol1`. 



# Recipe for running DeepSol (For all experiments reported in the manuscript)

Recipe is contained in the script `run.sh`. To see the options run `./run.sh` and you shall see the following:

```
main options (for others, see top of script file)
  --model (deepsol1/deepsol2/deepsol3)               # model architecture to use
  --mode  (preprocess/train/decode/cv)               # data preparation or decode or cross-validate using an existing model
  --stage (1/2)                                      # point to run the script from 
  --conf_file                                        # model parameter file
  --keras_backend                                    # backend for keras
  --cuda_root                                        # the path cuda installation
  --device (cuda/cpu)                                # device to use for running the recipe
  --test_file                                        # name of the new test file
```
There are two stages in the script. 

1. Data preparation as demonstrated with new test data. Prepared data is already provided in the data folder: `protein.data` and `protein_with_bio.data`.
2. Model building with `--mode train` and decoding with best DeepSol models using `--mode decode`. Information about `--mode cv` is given in "parameter variance check" section.

We provide support for gpu usage using the option `--device cuda`. More details in the GPU section.

## CPU

### Train New Models

Train DeepSol models using pre-compiled training, validation data and optimal hyper-parameter setting as in `parameters.json` file:

  1. `./run.sh --model deepsol1 --stage 2 --mode train --device cpu data/protein.data` 
  2. `./run.sh --model deepsol2 --stage 2 --mode train --device cpu data/protein_with_bio.data`

Result will be a model named `deepsol1` or `deepsol2` stored in `results/models/`. 

Note that we used `--model deepsol2`, you can use `deepsol3` for step 2. Ignore `UserWarning` at the output.

### Test Best DeepSol Models

Test existing DeepSol models with pre-compiled test data:

  1. `./run.sh --model deepsol1 --stage 2 --mode decode --device cpu data/protein.data`
  2. `./run.sh --model deepsol2 --stage 2 --mode decode --device cpu data/protein_with_bio.data` 

Result will be saved in `results/reports/`.

Note that we used `--model deepsol2`, you can use `deepsol3` for step 2. 


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

Note that we used `--model deepsol2`, you can use `deepsol3` for step 2. Ignore `UserWarning` at the output.

Also, `--cuda_root` should be the path to your cuda installation. By default it is `/usr/local/cuda`.

### Test Best DeepSol Models

Test existing DeepSol models with pre-compiled test data:

  1. `./run.sh --model deepsol1 --stage 2 --mode decode --cuda_root <path-to-your-cuda-installation> --device cuda data/protein.data`
  2. `./run.sh --model deepsol2 --stage 2 --mode decode --cuda_root <path-to-your-cuda-installation> --device cuda data/protein_with_bio.data`.

Result will be saved in `results/reports/`.

Note that we used `--model deepsol2`, you can use `deepsol3` for step 2. 


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

# FAQ

1. What all system can your code run on?

A) On most Linux based systems, we tested on Ubuntu 14.04 and 14.10, RedHat 7.4 Maipo and Arch (both cpu and gpu).

2. How to remove a conda environment?

A) conda remove --name dsol --all

3. What if I get error ` error while loading shared libraries: libmpfr.so.4:` while installing SCRATCH on Arch ?

A) Do `ln -s /usr/lib/libmpfr.so.6.0.0 /usr/lib/libmpfr.so.4`. SCRATCH looks for `mpfr.so.4` but Arch has a newer version, so we symlink the old location to the new library.
   


   


