import os
import argparse
import pickle
import json
import numpy as np
np.random.seed(1337)

import dsol.utils as utils
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='decoder.py')

parser.add_argument('-parameter_setting_id', required=True,
                    help='Which parameter setting from the configuration file to use')
parser.add_argument('-conf_file', required=True, type=str,
                            help='configuration json file from which to read the arguments')
parser.add_argument('-data', required=True,
                    help='Path to the *-.data file from preprocess.py')
parser.add_argument('-model', required=True,
                    help='Path to the best model for this configuration of DeepSol')
parser.add_argument('-maxlen', default=1200,
                    help='Maximum sequence length')
parser.add_argument('-vocab_size', type=int, default=23,
                    help='Number of input units')
parser.add_argument('-epochs', type=int, default=20,
                    help='Number of training epochs')
parser.add_argument('-patience', type=int, default=10,
                    help='Used for early stopping')
parser.add_argument('-results_dir', type=str, default='results',
                    help='Used for early stopping')

static_args = parser.parse_args()
static_args.parameters_json = json.load(open(static_args.conf_file, 'r'))
dynamic_args = static_args.parameters_json[static_args.parameter_setting_id]

dynamic_args['optim'] = dynamic_args['optim_config'].split(',')[0]
dynamic_args['learning_rate'] = float(dynamic_args['optim_config'].split(',')[1])

def get_model_path():
    model_name = static_args.model
    model_dir = static_args.results_dir+'/models/' 
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = model_dir + model_name
    return model_path


def get_classification_performance_path():
    report_name = 'report-cv-parameter_setting-' + str(static_args.parameter_setting_id)
    reports_dir = static_args.results_dir + '/reports/'
    if not os.path.isdir(reports_dir):
        os.makedirs(reports_dir)
    report_path = reports_dir + report_name
    return report_path


def save_classification_performance(string, report):
    save_at_path = get_classification_performance_path()
    with open(save_at_path, 'a') as f:
        f.write(string + '\n' + report + '\n')
    f.close()
    

def save_classification_prediction(prediction_class,prediction_prob):
    reports_dir = static_args.results_dir + '/reports/'
    report_path = reports_dir + 'report-cv-prediction-'+ str(static_args.parameter_setting_id) + '.txt'
    with open(report_path,'w') as f:
        f.write('Predicted_Class'+'\t'+'P0'+'\t'+'P1'+'\n');
        for i in range(0,len(prediction_class)):
            f.write(str(prediction_class[i])+'\t'+str(prediction_prob[i][0])+'\t'+str(prediction_prob[i][1])+'\n')
    f.close()


def load_data():
    dataset = pickle.load(open(static_args.data, 'rb'))
    return dataset


def get_probabilities(best_model, x):
    probabilities = best_model.predict(x,
                                       batch_size=int(dynamic_args['batch_size']))
    return probabilities


def get_classification_performance(best_model, x, y):
    pred_probs = get_probabilities(best_model, x)
    preds = pred_probs.argmax(axis=-1)
    y = y.argmax(axis=-1)
    acc = accuracy_score(y, preds)
    score_report = classification_report(y, preds)
    cm = confusion_matrix(y, preds)
    return [acc, score_report, cm, preds, pred_probs]

def get_classification_prediction(best_model, x, y):
    pred_probs = get_probabilities(best_model, x)
    preds = pred_probs.argmax(axis=-1)
    y = y.argmax(axis=-1)
    return [preds, pred_probs]


def main():
    data = load_data()
    #To perform 10-fold cross-validation
    kf = KFold(n_splits=10,shuffle=False)

    if (static_args.parameter_setting_id=="deepsol1"):
        x_train, y_train = data['train']['src'], data['train']['tgt']
        dynamic_args['num_classes'] = len(set(y_train))

        x_train = np.array(utils.pad_sequecnes(x_train, static_args.maxlen))
        y_train = utils.get_one_hot(y_train, dynamic_args['num_classes'])

        print('Training data: ', x_train.shape)

        x_val, y_val = data['valid']['src'], data['valid']['tgt']
        x_val = np.array(utils.pad_sequecnes(x_val, static_args.maxlen))
        y_val = utils.get_one_hot(y_val, dynamic_args['num_classes'])

        print('Valid data: ', x_val.shape)

        x_full = np.concatenate((x_train,x_val),axis=0)
        y_full = np.concatenate((y_train,y_val),axis=0)

        print('Full Train data: ', x_full.shape)

    else:
        x_train, x_train_bio, y_train = data['train']['src'], data['train']['src_bio'], data['train']['tgt']
        dynamic_args['num_classes'] = len(set(y_train))
        
        x_train = np.array(utils.pad_sequecnes(x_train, static_args.maxlen))
        x_train_bio = np.array(x_train_bio)[:, :-1]
        dynamic_args['num_bio_feats'] = int(x_train_bio.shape[1])
        
        y_train = utils.get_one_hot(y_train, dynamic_args['num_classes'])

        print('Training data: ', x_train.shape)
        print('Training data Bio: ', x_train_bio.shape)

        x_val, x_val_bio, y_val = data['valid']['src'], data['valid']['src_bio'], data['valid']['tgt']
        x_val = np.array(utils.pad_sequecnes(x_val, static_args.maxlen))
        x_val_bio = np.array(x_val_bio)[:, :-1]
        y_val = utils.get_one_hot(y_val, dynamic_args['num_classes'])
        
        print('Valid data: ', x_val.shape)
        print('Valid data Bio: ', x_val_bio.shape)

        x_full = np.concatenate((x_train,x_val),axis=0)
        x_full_bio = np.concatenate((x_train_bio,x_val_bio),axis=0)
        y_full = np.concatenate((y_train,y_val),axis=0)

        print('Full Train data: ',x_full.shape)
        print('Full Train Bio: ',x_full_bio.shape)

    best_model = utils.load_model(get_model_path())
    print('Loaded best model for ',static_args.parameter_setting_id)

    #Get file where results are to be saved
    results_filename_with_path = get_classification_performance_path()
    if os.path.exists(results_filename_with_path):
    	os.remove(results_filename_with_path)

    #Keep average scores for cross_validation
    acc_test_vec = []
    count = 1
    for train_index,test_index in kf.split(x_full):
        print('Starting CV Iteration: ',str(count))
        x_test = x_full[test_index]
        y_test = y_full[test_index]
        y_test = np.array(y_test, dtype='int32')

        if (static_args.parameter_setting_id=='deepsol1'):
            [acc_test, score_report_test, cm_test, pred_test, pred_prob_test] = get_classification_performance(best_model,x_test,y_test)
        else:
            x_test_bio = x_full_bio[test_index]
            [acc_test, score_report_test, cm_test, pred_test, pred_prob_test] = get_classification_performance(best_model,[x_test,x_test_bio],y_test)

        #Save output on disk
        save_classification_performance('Iteration: '+str(count),'')
        save_classification_performance('Test Accuracy: ', str(acc_test))
        save_classification_performance('Score Report Test: : ', str(score_report_test))
        save_classification_performance('Confusion Matrix test: ', str(cm_test))
        acc_test_vec.append(acc_test)
        count=count+1

    mean_acc = (1.0*sum(acc_test_vec))/len(acc_test_vec)
    save_classification_performance('Mean CV accuracy: ',str(mean_acc))

    print('Finished cross-validation')
    

if __name__ == '__main__':
    main()
