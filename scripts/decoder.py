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
    report_name = 'report-parameter_setting-' + str(static_args.parameter_setting_id)
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
    report_path = reports_dir + 'report-prediction-'+ str(static_args.parameter_setting_id) + '.txt'
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

    if ("deepsol1" in static_args.parameter_setting_id):
        x_train, y_train = data['train']['src'], data['train']['tgt']
        dynamic_args['num_classes'] = len(set(y_train))

        x_train = np.array(utils.pad_sequecnes(x_train, static_args.maxlen))
        y_train = utils.get_one_hot(y_train, dynamic_args['num_classes'])

        print('Training data: ', x_train.shape)

        x_test, y_test = data['test']['src'], data['test']['tgt']
        x_test = np.array(utils.pad_sequecnes(x_test, static_args.maxlen))
        y_test = utils.get_one_hot(y_test, dynamic_args['num_classes'])

        print('Test data: ', x_test.shape)

    else:
        x_train, x_train_bio, y_train = data['train']['src'], data['train']['src_bio'], data['train']['tgt']
        dynamic_args['num_classes'] = len(set(y_train))
        
        x_train = np.array(utils.pad_sequecnes(x_train, static_args.maxlen))
        x_train_bio = np.array(x_train_bio)[:,:-1] 
        dynamic_args['num_bio_feats'] = int(x_train_bio.shape[1])
        
        y_train = utils.get_one_hot(y_train, dynamic_args['num_classes'])

        print('Training data: ', x_train.shape)
        print('Training data Bio: ', x_train_bio.shape)
        print(x_train_bio[1])

        x_test, x_test_bio, y_test = data['test']['src'], data['test']['src_bio'], data['test']['tgt']
        x_test = np.array(utils.pad_sequecnes(x_test, static_args.maxlen))
        x_test_bio = np.array(x_test_bio)[:, :-1]
        y_test = utils.get_one_hot(y_test, dynamic_args['num_classes'])
        
        print('Test data: ', x_test.shape)
        print('Test data Bio: ', x_test_bio.shape)


    best_model = utils.load_model(get_model_path())

    if ("deepsol1" in static_args.parameter_setting_id):
        [pred_test,pred_prob_test] = get_classification_prediction(best_model,x_test,y_test)
    else:
        [pred_test,pred_prob_test] = get_classification_prediction(best_model,[x_test,x_test_bio],y_test)

    print('Finished testing')
    #save ion disk
    get_classification_performance_path()
    save_classification_prediction(pred_test,pred_prob_test)
    

if __name__ == '__main__':
    main()
