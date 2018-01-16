# for running DSOL1

conf_file=parameters.json
parameter_setting_id="DeepSol1"
keras_backend=theano
cuda_root='/usr/local/cuda'
device=cpu
model=
mode=
stage=0

. parse_options.sh || exit 1;


if [ $# != 1 ]; then
    echo "usage: ./run.sh <data-dir>"
    echo "e.g.:  ./run.sh data/protein_dsol1.data"
    echo "main options (for others, see top of script file)"
    echo "  --model                                            # default full."
    echo "  --mode                                             # default full."
    echo "  --stage                                             # default full."
    echo "  --conf_file                                        # default full."
    echo "  --parameter_setting_id                             # default full."
    echo "  --keras_backend                                    # use graphs in src-dir"
    echo "  --cuda_root                                        # number of parallel jobs"
    echo "  --device                                           # config containing options"
    exit 1;
fi

data=$1

if [[ $device == 'cpu' ]] ; then
        export THEANO_FLAGS="base_compiledir=$(pwd)/.theano,device=${device},floatX=float32"
fi

if [[ $device == *"cuda"* ]]; then
    export CUDA_HOME=${cuda_root}
    export LD_LIBRARY_PATH=${cuda_root}/lib64
    export THEANO_FLAGS="base_compiledir=$(pwd)/.theano,cuda.root=${cuda_root},device=${device},floatX=float32"
fi

if [ $stage -le 0 ] ; then
    
    if [[ $model == "deepsol1" ]]; then
	KERAS_BACKEND=${keras_backend} python preprocess_dsol1.py -train_src data/train_src -train_tgt data/train_tgt -valid_src data/val_src -valid_tgt data/val_tgt -test_src data/test_src -test_tgt data/test_tgt -save_data data/protein
    fi
fi


if [ $stage -le 1 ] ; then
    
    if [[ "$mode" = "train" ]] ; then

	if [[ $model == "deepsol1" ]]; then
	    KERAS_BACKEND=${keras_backend} python main_dsol1.py -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
	fi
    fi
    
    if [[ "$mode" == "decode" ]] ; then
	if [[ $model == "deepsol1" ]]; then
	    KERAS_BACKEND=${keras_backend} python main_dsol1.py -conf_file ${conf_file} -parameter_setting_id ${parameter_setting_id} -data ${data}
	fi
    fi
fi

# for running DSOL2
#python main_dsol2.py -data data/protein_dsol2.data -conf_file parameters.json -parameter_setting_id DeepSol2

# for running DSOL3 
#python main_dsol3.py -data data/protein_dsol2.data -conf_file parameters.json -parameter_setting_id DeepSol3
