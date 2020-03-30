# Modified version of https://github.com/Jeff09/Word-Sense-Disambiguation-using-Bidirectional-LSTM
# Modified by Bleau Moores, Lisa Ewen, Tim Heydrich
# Last Modified: 27/03/2020 by Tim Heydrich

from data import *
from glove import *
from model import *
from test_anser_loader import *
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K
from scipy import spatial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config
train_path = '/data/senseval2/eng-lex-sample.training.xml'
test_path = '/data/senseval2/eng-lex-samp.evaluation.xml'

# load data
train_data_ = load_train_data(23)
test_data = load_test_data(23)
print('Dataset size (train/test): %d / %d' % (len(train_data_), len(test_data)))

EMBEDDING_DIM = 100
print('Embedding vector: %d' % EMBEDDING_DIM)

'''PREPARING TRAINING DATA'''

print("Preparing Training Data")
word_to_id = build_vocab(train_data_)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data_)
print('Vocabulary size: %d' % len(word_to_id))
train_target_sense_to_context = build_context(train_data_, word_to_id)

embedding_matrix = fill_with_gloves(word_to_id, 100)

target_sense_to_context_embedding = build_embedding(train_target_sense_to_context, embedding_matrix, len(word_to_id),
                                                    EMBEDDING_DIM)

train_ndata = convert_to_numeric(train_data_, word_to_id, target_word_to_id, target_sense_to_id,
                                 n_senses_from_target_id, target_sense_to_context_embedding, is_training=True)
print("Preparing Test Data")
sense_embeddings_ = get_embedding(sense_embedding_file)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id,
                                sense_embeddings_, is_training=False)
n_step_f = 40
n_step_b = 40
print('n_step forward/backward: %d / %d' % (n_step_f, n_step_b))
MAX_SEQUENCE_LENGTH = 40


# ======================================================================================================================
# ======================================================================================================================

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


# ======================================================================================================================

def cos_sim(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1, keepdims=True)


# ======================================================================================================================

def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# ======================================================================================================================

def own_model(train_forward_data, train_backward_data, train_sense_embedding, test_f, test_b, test_i,
              val_forward_data=None, val_backward_data=None, val_sense_embedding=None,
              n_units=100, dense_units=256, is_training=True, EMBEDDING_DIM=100, epochs=100, batch_size=2048,
              init_word_vecs=None, ):
    model = get_model(n_units=n_units, dense_unints=dense_units, is_training=is_training, emb_dim=EMBEDDING_DIM,
                      init_word_vecs=init_word_vecs, max_sequence_length=40, word_to_id=word_to_id)

    # Switchable optimizers
    opti = optimizers.Nadam(clipnorm=1.)  # , clipvalue=0.5
    # opti = optimizers.SGD(lr=0.00001, momentum=0.1)
    # opti = optimizers.Adam(lr=0.00001)

    model.compile(loss='mse', optimizer=opti, metrics=[cos_distance, get_f1])

    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    bst_model_path = "weights.best.hdf5"
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)

    hist = model.fit([train_forward_data, train_backward_data], train_sense_embedding,
                     validation_data=([val_forward_data, val_backward_data], val_sense_embedding),
                     epochs=epochs, batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])

    model.save('1_project_2_TT.h5')

    def get_embedded_sense(goal_key):
        for elem in train_data_:
            key = elem['target_sense']
            if key in goal_key:
                return elem['id']
        return -1

    ''' Modified testing Code '''
    # Uses the testing target sense id to get the actual embedding from target_sense_to_context_embedding
    # That actual embedding is then used to calculate the cosine distance between it and the predicted vector
    pred_a = model.predict([test_f, test_b])
    cos_sim_total = 0
    counter = 0
    not_testable = 0
    test_answers = get_test_ansers(23)
    for i in range(len(pred_a)):
        pred = pred_a[i]
        goal_id = test_i[i]
        idx = test_answers.index[test_answers['Targets'] == goal_id]
        # This is for the entries where the sense was either just 'U' or 'P' or both
        if len(idx) == 0:
            continue
        goal_key = test_answers.iloc[idx]['Senses'].to_numpy()[0]
        train_id_key = get_embedded_sense(goal_key)
        # This is in case the testing target sense is not in the training corpus
        if train_id_key == -1:
            not_testable += 1
            continue
        goal_embedding = target_sense_to_context_embedding.get(train_id_key)
        cos_sim = (1 - spatial.distance.cosine(goal_embedding, pred))
        cos_sim_total += cos_sim
        counter += 1

    print("Average Testing Cos Sim:", (cos_sim_total / counter))
    print("Number of untestable due to the lack of a comparable embedding:", not_testable)


# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    grouped_by_target = group_by_target(train_ndata)
    train_data, val_data = split_grouped(grouped_by_target, 0)

    # Getting test data, needed seperate method as get_data does not work straight from github
    test_grouped_by_target = group_by_target(test_ndata)
    test_data_, _ = split_grouped(test_grouped_by_target, 0)
    test_forward_data, test_backward_data, test_target_sense_ids = get_data_test(test_data_, n_step_f, n_step_b)

    init_emb = fill_with_gloves(word_to_id, EMBEDDING_DIM)

    train_forward_data, train_backward_data, train_target_sense_ids, train_sense_embedding = get_data(train_data,
                                                                                                      n_step_f,
                                                                                                      n_step_b)
    val_forward_data, val_backward_data, val_target_sense_ids, val_sense_embedding = get_data(val_data, n_step_f,
                                                                                              n_step_b)
    own_model(train_forward_data, train_backward_data, train_sense_embedding, test_forward_data, test_backward_data,
              test_target_sense_ids,
              val_forward_data, val_backward_data, val_sense_embedding,
              init_word_vecs=init_emb, epochs=10)
