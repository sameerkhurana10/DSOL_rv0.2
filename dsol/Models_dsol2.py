"""
"""
import keras
from keras.preprocessing import sequence
from keras.layers import merge, Embedding, Bidirectional, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.core import *
from keras.models import *
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import GRU, LSTM

class DeepSol():
    def __init__(self, static_args, dynamic_args):
        """
        """
        self.maxlen = int(static_args.maxlen)
        self.vocab_size = int(static_args.vocab_size)
        self.embedding_dim = int(dynamic_args['embedding_dim'])
        self.em_drop = float(dynamic_args['em_drop'])
        self.num_classes = int(dynamic_args['num_classes'])
        self.fc_config = [conf.split(',') for conf in dynamic_args['fc_config'].split('-')]
        self.fc_dims = [f[0] for f in self.fc_config]
        self.fc_drop = [f[1] for f in self.fc_config]
        
        self.cnn_config = dynamic_args['cnn_config']
        if self.cnn_config is not None:
            self.cnn_config = [conf.split(',') for conf in dynamic_args['cnn_config'].split('-')]
            self.kernel_sizes = [f[0] for f in self.cnn_config]
            self.feature_maps = [f[1] for f in self.cnn_config]
            self.max_pool_size = [f[2] for f in self.cnn_config]

        self.rnn_config = dynamic_args['rnn_config']
        if self.rnn_config is not None:
            self.rnn_config = dynamic_args['rnn_config'].split('-')
            self.rnn_bidirectional = dynamic_args['rnn_bidirectional']

        self.attention = dynamic_args['attention']
        self.biofeats = dynamic_args['biofeats']
        self.protein_seq_feats = dynamic_args['protein_seq_feats']
        self.num_bio_feats = int(dynamic_args['num_bio_feats'])

    def fetch_model_def(self):

        input_protein_seq = Input(shape=(self.maxlen,))
        input_bio_feats = Input(shape=(self.num_bio_feats,))
        embedding_layer = Embedding(self.vocab_size,
                                    self.embedding_dim,
                                    input_length=self.maxlen,
                                    )
        # Embed input to a continuous space
        model = None
        if self.protein_seq_feats is not None:
            embedded = embedding_layer(input_protein_seq)
            embedded = SpatialDropout1D(self.em_drop)(embedded)
            # Attention after embedding
            if self.attention is not None:
                embedded = utils.attention_3d_block(embedding, embedded)

                # Convolution Layers
            x = embedded
            if self.cnn_config is not None:
                for k in range(len(self.cnn_config)):
                    convs = []
                    # Lists of feature maps and kernel sizes
                    # Check to see if we have more than one kernel
                    if ':' in self.feature_maps[k]:
                        feature_maps = [int(f) for f in self.feature_maps[k].split(':')]
                        kernel_sizes = [int(f) for f in self.kernel_sizes[k].split(':')]
                    else:
                        feature_maps = [int(self.feature_maps[k])]
                        kernel_sizes = [int(self.kernel_sizes[k])]
                    # An integer or a string
                    pool_size = self.max_pool_size[k]
                    for i in range(len(feature_maps)):
                        if k == len(self.cnn_config)-1 and self.rnn_config is None:
                            conv_layer = Conv1D(filters=feature_maps[i],
                                                kernel_size=kernel_sizes[i],
                                                padding='valid',
                                                activation='relu',
                                                strides=1)
                        else:
                            conv_layer = Conv1D(filters=feature_maps[i],
                                                kernel_size=kernel_sizes[i],
                                                padding='same',
                                                activation='relu',
                                                strides=1)
                        
                        conv_out = conv_layer(x)
                        # Global can't be given as pool_size if RNN nis to be used after CNN. Need to be careful
                        if not pool_size.lower() == 'no':
                            if pool_size.lower() == 'global' and self.rnn_config is None:
                                conv_out = MaxPooling1D(pool_size=conv_layer.output_shape[1])(conv_out)
                                
                                conv_out = Flatten()(conv_out)
                            else:
                                conv_out = MaxPooling1D(pool_size=int(pool_size))(conv_out)
                                
        
                        convs.append(conv_out)
    
                    # the below concat approach will be deprecated in 4 months
                    x = merge(convs, mode='concat') if len(convs) > 1 else convs[0]

            # Recurrent Layers
            if self.rnn_config is not None:
                for j in range(len(self.rnn_config)):
                    rnn_unit_type = self.rnn_config[j].split(',')[0]
                    rnn_units = int(self.rnn_config[j].split(',')[1])
                    rnn_drop = float(self.rnn_config[j].split(',')[2])
                    rnn_rec_drop = float(self.rnn_config[j].split(',')[3])
                    #rnn_out_drop = float(self.rnn_config[j].split(',')[4])
    
                    if rnn_unit_type.lower() == 'gru':
                        if j == len(self.rnn_config)-1:
                            if self.rnn_bidirectional is None:
                                x = GRU(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop)(x)
                            else:
                                x = Bidirectional(GRU(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop))(x)
                        else:
                            if self.rnn_bidirectional is None:
                                x = GRU(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop, return_sequences=True)(x)
                            else:
                                x = Bidirectional(GRU(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop, return_sequences=True))(x)
                            
                    elif rnn_unit_type.lower() == 'lstm':
                        if j == len(self.rnn_config)-1:
                            if self.rnn_bidirectional is None:
                                x = LSTM(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop)(x)
                            else:
                                x = Bidirectional(LSTM(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop))(x)
                        else:
                            if self.rnn_bidirectional is None:
                                x = LSTM(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop, return_sequences=True)(x)
                            else:
                                x = Bidirectional(LSTM(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop, return_sequences=True))(x)
                            
            # Append biological feats
            if self.biofeats is not None:
                x = merge([x, input_bio_feats], mode='concat')
    
            # Dense Layers
            for l in range(len(self.fc_config)):
                fc_dim = int(self.fc_dims[l])
                fc_dropout = float(self.fc_drop[l])
                x = Dense(fc_dim)(x)
                x = Dropout(fc_dropout)(x)
                x = Activation('relu')(x)
            main_output = Dense(self.num_classes, activation='sigmoid')(x)
            
            if self.biofeats is not None:
                model = Model(input=[input_protein_seq, input_bio_feats], output=main_output)
            else:
                model = Model(input=input_protein_seq, output=main_output)
        elif self.biofeats is not None:
            # this will run if with only biological feats
            x = input_bio_feats
            for l in range(len(self.fc_config)):
                fc_dim = int(self.fc_dims[l])
                fc_dropout = float(self.fc_drop[l])
                x = Dense(fc_dim)(x)
                x = Dropout(fc_dropout)(x)
                x = Activation('relu')(x)
            main_output = Dense(self.num_classes, activation='sigmoid')(x)
            model = Model(input=input_bio_feats, output=main_output)

        return model
