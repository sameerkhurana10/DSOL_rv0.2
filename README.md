# DeepSol: A Deep Learning Framework for Sequence-Based Protein Solubility Prediction

![alt text](http://people.csail.mit.edu/sameerk/dsol.svg)

# Setting up the environment:

## Install Anaconda
1. Download Anaconda installer python3.* for linux : https://www.anaconda.com/download/#linux
2. Run the installer : `bash Anaconda3-5.0.1-Linux-x86_64.sh` and follow the instructions to install anaconda at your preferred location

## Activate environment
1. Run `export PATH=<your_anaconda_folder>/bin:$PATH`
2. Run `conda env create -f environment.yml`
3. Run `source activate dsol`


# Recipe for running DeepSol:

## Data Preparation

1. For DeepSol1: `./run.sh --model deepsol1 --stage 1 --mode prep data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 1 --mode prep --device cuda0 data/protein.data`

2. For DeepSol2: `./run.sh --model deepsol2 --stage 1 --mode prep data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 1 --mode prep --device cuda0 data/protein_with_bio.data`

## Model Training
### CPU only
Run `./run.sh --model deepsol1 --stage 2 --mode prep data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 1 --mode prep data/protein.data` 
 
If you want to only predict on test set you can give the argument `--mode decode` with `--stage 2` for just decoding, while for both training and decoding you can give the arguement `--mode all` with `--stage 2`.

2. Recipe for Decoding only:

For CPU simply run `./run.sh --model deepsol1 --stage 2 --mode decode data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 2 --mode decode --device cuda0 data/protein.data`

3. Recipe for Training and Decoding:

For CPU simply run `./run.sh --model deepsol1 --stage 2 --mode all data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 2 --mode all --device cuda0 data/protein.data`

4. Recipe for Cross-Validation:

For CPU simply run `./run.sh --model deepsol1 --stage 3 --mode cv data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 3 --mode cv --device cuda0 data/protein.data`

