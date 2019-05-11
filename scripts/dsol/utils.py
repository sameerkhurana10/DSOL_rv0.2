import keras
from keras.preprocessing import sequence
from keras.layers import Concatenate, Multiply, Bidirectional, TimeDistributed, concatenate, multiply
from keras.layers.core import *
from keras.models import *


def get_input_layer(length):
    return keras.layers.Input(shape=(length,))


def get_embedding_layer(input_size, output_size, input_length):
    return keras.layers.Embedding(input_size, output_size,
                                  input_length=input_length)


def get_conv1D_layer(out_channels, kernel_size, border_mode='valid',
                     activation='relu', kernel_regularizer=None,
                     bias_regularizer=None, activity_regularizer=None):
    return keras.layers.convolutional.Conv1D(filters=out_channels,
                                             kernel_size=kernel_size,
                                             padding=border_mode,
                                             activation=activation,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             strides=1)


def get_GRU_layer(units, dropout=0.0, recurrent_dropout=0.0,
                  bidirectional=None, return_sequences=False):
    if not bidirectional:
        return keras.layers.recurrent.GRU(
                                      units, activation='tanh',
                                      recurrent_activation='hard_sigmoid',
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      recurrent_initializer='orthogonal',
                                      bias_initializer='zeros',
                                      kernel_regularizer=None,
                                      recurrent_regularizer=None,
                                      bias_regularizer=None,
                                      activity_regularizer=None,
                                      kernel_constraint=None,
                                      recurrent_constraint=None,
                                      bias_constraint=None,
                                      dropout=dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=return_sequences
            )
    else:
        return Bidirectional(keras.layers.recurrent.GRU(
                                    units, activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    return_sequences=return_sequences
                                    ))


def get_LSTM_layer(units, dropout=0.0, recurrent_dropout=0.0,
                  bidirectional=None, return_sequences=False):
    if not bidirectional:
        return keras.layers.recurrent.LSTM(
                                      units, activation='tanh',
                                      recurrent_activation='hard_sigmoid',
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      recurrent_initializer='orthogonal',
                                      bias_initializer='zeros',
                                      kernel_regularizer=None,
                                      recurrent_regularizer=None,
                                      bias_regularizer=None,
                                      activity_regularizer=None,
                                      kernel_constraint=None,
                                      recurrent_constraint=None,
                                      bias_constraint=None,
                                      dropout=dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      return_sequences=return_sequences
            )
    else:
        return Bidirectional(keras.layers.recurrent.LSTM(
                                    units, activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    return_sequences=return_sequences
                                    ))


def get_spatial_dropout1D(p=0.2):
    return keras.layers.core.SpatialDropout1D(p)


def get_max_pooling1D_layer(pool_size, strides=None, padding='valid'):
    return keras.layers.pooling.MaxPooling1D(pool_size=pool_size,
                                             strides=strides,
                                             padding=padding)


def get_mean_pooling1D_layer(pool_size, strides=None, padding='valid'):
    return keras.layers.pooling.AveragePooling1D(pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding)


def get_global_max_pooling1D_layer():
    return keras.layers.pooling.GlobalMaxPooling1D()


def get_flatten_layer():
    return keras.layers.core.Flatten()


def get_drop_out_layer(rate, noise_shape=None, seed=None):
    return keras.layers.core.Dropout(rate, noise_shape=noise_shape, seed=seed)


def get_masking_layer(mask_value):
    return keras.layers.core.Masking(mask_value=mask_value)


def get_adam_optim(lr=0.001):
    return keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                    decay=0.0)


def get_rmsprop_optim(lr=0.001):
    return keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)


def get_sgd_optim(lr=0.01, nesterov=False):
    return keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=nesterov)


def get_dense_layer(units, activation):
    return keras.layers.core.Dense(units, activation=None)


def get_activation_layer(activation):
    return keras.layers.core.Activation(activation)


#def get_merge_concat_layer(list_tensors, mode='concat'):
#    return (list_tensors, mode=mode)


#def get_merge_add_layer():
#    return keras.layers.merge.Add()


#def get_merge_multiply_layer():
#    return keras.layers.merge.Multiply()


def get_model(main_input, main_output):
    return keras.models.Model(inputs=main_input, outputs=main_output)


def get_model_checkpoint(model_path, verbose=1, save_best_only=True):
    return keras.callbacks.ModelCheckpoint(filepath=model_path,
                                           verbose=verbose,
                                           save_best_only=save_best_only)


def get_early_stopping_cbk(monitor='val_loss', patience=5):
    return keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)


def load_model(filepath):
    return keras.models.load_model(filepath)


def pad_sequecnes(sequence_list, maxlen):
    return sequence.pad_sequences(sequence_list, maxlen)


def get_one_hot(labels_list, num_classes):
    return keras.utils.np_utils.to_categorical(labels_list, num_classes)


def attention_3d_block(input_layer, inputs):
    # Figure this out later
    SINGLE_ATTENTION_VECTOR = False
    # Code taken from: https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
    input_dim = int(input_layer.output_shape[2])
    TIME_STEPS = int(input_layer.output_shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = TimeDistributed(Dense(TIME_STEPS, activation='softmax'))(a)
    # a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul
