#!/bin/bash
# Copyright 2018  Massachusetts Institute of Technology (Sameer Khurana)
#                 Qatar Computing Research Institute    (Raghavendra Mall)
# Apache 2.0

# Recipe for running DeepSol training and decoding

# Begin configuration section.

conf_file=parameters.json
keras_backend=theano
cuda_root='/usr/local/cuda'
device=cpu
model=
mode=
stage=0

echo "$0 $@"  # Print the command line for logging

. parse_options.sh || exit 1;


if [ $# != 1 ]; then
    echo "usage: ./run.sh <options> data_file"
    echo "e.g.:  ./run.sh data/protein.data"
    echo "main options (for others, see top of script file)"
    echo "  --model (deepsol1/deepsol2|deepsol3)               # model architecture to use"
    echo "  --mode  (train/decode)                             # train up a new model or use an existing model"
    echo "  --stage (1/2)                                      # point to run the script from "
    echo "  --conf_file                                        # model parameter file"
    echo "  --keras_backend                                    # backend for keras"
    echo "  --cuda_root                                        # the path cuda installation"
    echo "  --device (cuda/cpu)                                # device to use for running the recipe"
    exit 1;
fi

data=$1
parameter_setting_id=$model

if [[ $device == 'cpu' ]] ; then
        export THEANO_FLAGS="base_compiledir=$(pwd)/.theano,device=${device},floatX=float32"
fi

if [[ $device == *"cuda"* ]]; then
    export CUDA_HOME=${cuda_root}
    export LD_LIBRARY_PATH=${cuda_root}/lib64
    export THEANO_FLAGS="base_compiledir=$(pwd)/.theano,cuda.root=${cuda_root},device=${device},floatX=float32"
fi



if [ $stage -le 1 ] ; then
    echo "++++++++++++++++++++ DATA PREPARATION ++++++++++++++++++++"
    if [[ $model == "deepsol1" ]]; then
	KERAS_BACKEND=${keras_backend} python preprocess_dsol1.py -train_src data/train_src -train_tgt data/train_tgt -valid_src data/val_src -valid_tgt data/val_tgt -test_src data/test_src -test_tgt data/test_tgt -save_data ${data}
    fi

    if [[ $model == "deepsol2" || $model == "deepsol3" ]]; then
	KERAS_BACKEND=${keras_backend} python preprocess_dsol2.py -train_src data/train_src -train_src_bio data/train_src_bio -train_tgt data/train_tgt -valid_src data/val_src -valid_src_bio data/val_src_bio -valid_tgt data/val_tgt -test_src data/test_src -test_src_bio data/test_src_bio -test_tgt data/test_tgt -save_data ${data}
    fi
fi


if [ $stage -le 2 ] ; then
    
    if [[ "$mode" == "train" ]] ; then

	echo "++++++++++++++++++++ BUILDING NEW MODELS ++++++++++++++++++++"

	if [[ $model == "deepsol1" ]]; then
	    KERAS_BACKEND=${keras_backend} python main_dsol1.py -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
	fi

	if [[ $model == "deepsol2" ]]; then
	    KERAS_BACKEND=${keras_backend} python main_dsol2.py -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
	fi

	if [[ $model == "deepsol3" ]]; then
	    KERAS_BACKEND=${keras_backend} python main_dsol1.py -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
	fi
    fi
    
    if [[ "$mode" == "decode" ]] ; then

	echo "++++++++++++++++++++ DECODING USING EXISTING MODELS ++++++++++++++++++++"
	
	KERAS_BACKEND=${keras_backend} python decoder.py -model ${model} -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
    fi

    if [[ "$mode" == "cv" ]]; then
    	KERAS_BACKEND=${keras_backend} python cross_validation.py -model ${model} -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
    fi

fi

