Install Anaconda:

1. download Anaconda installer python3.* for linux : https://www.anaconda.com/download/#linux
2. run the installer : `bash Anaconda3-5.0.1-Linux-x86_64.sh` and follow the instructions to install anaconda at your preferred location
2. run `export PATH=<your_anaconda_folder>/bin:$PATH`
3. run `conda env create -f environment.yml`
4. activate the environment by running `source activate dsol`


run the code using pre-defined parameters:

1. Data preparation: `python preprocess_dsol1.py -train_src data/train_src -train_tgt data/train_tgt -valid_src data/val_src -valid_tgt data/val_tgt -test_src data/test_src -test_tgt data/test_tgt -save_data data/protein`

This will create a file protein.data in data folder. It is already provide for you.

2. for CPU simply run `./run.sh --model deepsol1 data/protein.data` and for GPU: run `./run.sh --model deepsol1 --device cuda0  data/protein.data`

    
