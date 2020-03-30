# Modified version of https://github.com/Jeff09/Word-Sense-Disambiguation-using-Bidirectional-LSTM
# Modified by Bleau Moores, Lisa Ewen, Tim Heydrich
# Last Modified: 27/03/2020 by Tim Heydrich

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization

# ======================================================================================================================
# ======================================================================================================================

def get_model(n_units=100, dense_unints=256, is_training=True, emb_dim=100,
              init_word_vecs=None, max_sequence_length=40, word_to_id=None):
    embedding_layer = Embedding(len(word_to_id),
                                emb_dim,
                                weights=[init_word_vecs],
                                input_length=max_sequence_length,
                                trainable=False)
    lstm_layer = LSTM(n_units, dropout=0.2, recurrent_dropout=0.2)
    # Use the one below when using model 2
    # lstm_layer = LSTM(n_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)

    forward_input = Input(shape=(max_sequence_length,), dtype='int32', name='forward_input')
    embedded_forward = embedding_layer(forward_input)

    # Comment out to use Basemodel
    forward_cnn = Conv1D(256, 3, activation='relu')(embedded_forward)
    forward_cnn2 = Conv1D(512, 3, activation='relu')(forward_cnn)
    forward_cnn3 = Conv1D(1024, 3, activation='relu')(forward_cnn2)
    forward_lstm = lstm_layer(forward_cnn3)

    #forward_lstm = lstm_layer(embedded_forward)


    backward_input = Input(shape=(max_sequence_length,), dtype='int32', name='backward_input')
    embedded_backward = embedding_layer(backward_input)

    #Comment out to use Basemodel
    backward_cnn = Conv1D(256, 3, activation='relu')(embedded_backward)
    backward_cnn2 = Conv1D(512, 3, activation='relu')(backward_cnn)
    backward_cnn3 = Conv1D(1024, 3, activation='relu')(backward_cnn2)
    backward_lstm = lstm_layer(backward_cnn3)

    #backward_lstm = lstm_layer(embedded_backward)

    merged = concatenate([forward_lstm, backward_lstm])

    #To be included when one wants to use model 2
    #merged_cnn = Conv1D(2048, 3, activation='relu')(merged)
    #merged_cnn2 = Conv1D(1024, 3, activation='relu')(merged_cnn)
    #merged_cnn3 = Conv1D(1024, 3, activation='relu')(merged_cnn2)
    #merged = Dropout(0.2)(merged_cnn3)

    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(units=dense_unints, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(emb_dim)(merged)
    model = Model(inputs=[forward_input, backward_input], outputs=preds)
    return model

# ======================================================================================================================
# ======================================================================================================================
