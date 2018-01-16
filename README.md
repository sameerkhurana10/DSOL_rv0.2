Install Anaconda:

1. download Anaconda installer python3.* for linux : https://www.anaconda.com/download/#linux
2. run the installer : `bash Anaconda3-5.0.1-Linux-x86_64.sh` and follow the instructions to install anaconda at your preferred location
2. run `export PATH=<your_anaconda_folder>/bin:$PATH`
3. run `conda env create -f environment.yml`
4. activate the environment by running `source activate dsol`


run the code using pre-defined parameters:


1. Recipe Training:

for CPU simply run `./run.sh --model deepsol1 --stage 1 --mode prep data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 1 --mode prep --device cuda0 data/protein.data`

stage 1 is data preparation and stage 2 is training or decoding. 
If you have prepared data you can give the argument stage 2 to move to training or decoding

3. Recipe for Decoding only:

for CPU simply run `./run.sh --model deepsol1 --stage 2 --mode decode data/protein.data` and for GPU: run `./run.sh --model deepsol1 --stage 2 --mode decode --device cuda0 data/protein.data`
    
