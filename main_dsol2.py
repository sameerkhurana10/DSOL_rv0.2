import os
import argparse
import pickle
import json
import numpy as np
np.random.seed(1337)

import dsol.utils as utils
import dsol.Models_dsol2 as Models
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='main.py')

parser.add_argument('-conf_file', required=True, type=str,
                    help='configuration json file from which to read the arguments')
parser.add_argument('-parameters_json', type=json.load,
                    help='configuration json file from which to read the arguments')
parser.add_argument('-parameter_setting_id', required=True,
                    help='Which parameter setting from the configuration file to use')
parser.add_argument('-data', required=True,
                    help='Path to the *-.data file from preprocess.py')
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

print(dynamic_args)


def get_model_path():
    model_name = 'model-parameter_setting-' + str(static_args.parameter_setting_id)
    model_dir = static_args.results_dir + '/models/'
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
    

def save_classification_prediction(prediction):
    reports_dir = static_args.results_dir + '/reports/'
    report_path = reports_dir + 'report-prediction-'+ str(static_args.parameter_setting_id) + '.txt'
    with open(report_path,'w') as f:
        for i in range(0,len(prediction)):
            f.write(str(prediction[i])+'\n')
    f.close()

def get_callbacks():
    stopping = utils.get_early_stopping_cbk(monitor='val_loss', patience=static_args.patience)
    model_path = get_model_path()
    checkpointer = utils.get_model_checkpoint(model_path,
                                              verbose=1,
                                              save_best_only=True)
    return [stopping, checkpointer]


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
#    print(y)
    acc = accuracy_score(y, preds)
    score_report = classification_report(y, preds)
    cm = confusion_matrix(y, preds)
    return [acc, score_report, cm, pred_probs]


def get_optimizer():
    if dynamic_args['optim'].lower() == 'adam':
        return utils.get_adam_optim(float(dynamic_args['learning_rate']))
    elif dynamic_args['optim'].lower() =='rmsprop':
        return utils.get_rmsprop_optim(int(dynamic_args['learning_rate']))
    elif dynamic_args['optim'].lower() =='sgd':
        return utils.get_sgd_optim(int(dynamic_args['learning_rate']))


def main():
    data = load_data()
    x_train, x_train_bio, y_train = data['train']['src'], data['train']['src_bio'], data['train']['tgt']
    dynamic_args['num_classes'] = len(set(y_train))

    x_train = np.array(utils.pad_sequecnes(x_train, static_args.maxlen))
    x_train_bio = np.array(x_train_bio)[:, :-1]
    dynamic_args['num_bio_feats'] = int(x_train_bio.shape[1])
    
    y_train = utils.get_one_hot(y_train, dynamic_args['num_classes'])

    print('Training data: ', x_train.shape)
    print('Training data Bio: ', x_train_bio.shape)
    print(x_train_bio[1])

    x_val, x_val_bio, y_val = data['valid']['src'], data['valid']['src_bio'], data['valid']['tgt']
    x_val = np.array(utils.pad_sequecnes(x_val, static_args.maxlen))
    x_val_bio = np.array(x_val_bio)[:, :-1]
    y_val = utils.get_one_hot(y_val, dynamic_args['num_classes'])

    print('Validation data: ', x_val.shape)
    print('Validation data Bio: ', x_val_bio.shape)

    x_test, x_test_bio, y_test = data['test']['src'], data['test']['src_bio'], data['test']['tgt']
    x_test = np.array(utils.pad_sequecnes(x_test, static_args.maxlen))
    x_test_bio = np.array(x_test_bio)[:, :-1]
    y_test = utils.get_one_hot(y_test, dynamic_args['num_classes'])

    print('Test data: ', x_test.shape)
    print('Test data Bio: ', x_test_bio.shape)
    
    model = Models.DeepSol(static_args, dynamic_args).fetch_model_def()

    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(),
                  metrics=['accuracy'])
    print(model.summary())

    # Training
    # Either use both bio and protein feats or just one of them
    acc_val, score_report_val, cm_val, pred_val = None, None, None, None
    if dynamic_args['biofeats'] is not None and dynamic_args['protein_seq_feats'] is not None:
        model.fit([x_train, x_train_bio], y_train, batch_size=dynamic_args['batch_size'],
                epochs=int(static_args.epochs),
                validation_data=([x_val, x_val_bio], y_val),
                callbacks=get_callbacks())
        best_model = utils.load_model(get_model_path())
        [acc_val, score_report_val, cm_val, pred_val] = get_classification_performance(best_model,
                                                                                        [x_val, x_val_bio],
                                                                                        y_val)
        [acc_test, score_report_test, cm_test, pred_test] = get_classification_performance(best_model,
                                                                                        [x_test,x_test_bio],
                                                                                        y_test)
    elif dynamic_args['protein_seq_feats'] is not None:
        model.fit(x_train, y_train, batch_size=dynamic_args['batch_size'],
                    epochs=int(static_args.epochs),
                    validation_data=(x_val, y_val),
                    callbacks=get_callbacks())
        best_model = utils.load_model(get_model_path())
        [acc_val, score_report_val, cm_val, pred_val] = get_classification_performance(best_model,
                                                                                        x_val,
                                                                                        y_val)
        [acc_test, score_report_test, cm_test, pred_test] = get_classification_performance(best_model,
                                                                                        x_test,
                                                                                        y_test)
    elif dynamic_args['biofeats'] is not None:
         model.fit(x_train_bio, y_train, batch_size=dynamic_args['batch_size'],
                    epochs=int(static_args.epochs),
                    validation_data=(x_val_bio, y_val),
                    callbacks=get_callbacks())
         best_model = utils.load_model(get_model_path())
         [acc_val, score_report_val, cm_val, pred_val] = get_classification_performance(best_model,
                                                                                        x_val_bio,
                                                                                        y_val)
         [acc_test, score_report_test, cm_test, pred_test] = get_classification_performance(best_model,
                                                                                        x_test_bio,
                                                                                        y_test)

    results_filename_with_path = get_classification_performance_path()
    if os.path.exists(results_filename_with_path):
    	os.remove(results_filename_with_path)

    # save on disk
    save_classification_performance('Validation Accuracy: ', str(acc_val))
    save_classification_performance('Test Accuracy: ', str(acc_test))
    save_classification_performance('Score Report Test: : ',
            str(score_report_test))
    save_classification_performance('Confusion Matrix test: ', str(cm_test))
    save_classification_prediction(pred_test)
    


if __name__ == '__main__':
    main()
