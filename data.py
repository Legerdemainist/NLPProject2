# Modified version of https://github.com/Jeff09/Word-Sense-Disambiguation-using-Bidirectional-LSTM
# Modified by Bleau Moores, Lisa Ewen, Tim Heydrich
# Last Modified: 27/03/2020 by Tim Heydrich

import lxml.etree as et
import collections
import re
from itertools import groupby
from glove import *
import csv

random.seed(0)

train_path2 = './data/senseval2/eng-lex-sample.training.xml'
test_path2 = './data/senseval2/eng-lex-samp.evaluation.xml'
train_path3 = './data/senseval3/EnglishLS.train.mod'
test_path3 = './data/senseval3/EnglishLS.test.mod'

senseval_key = './data/senseval2/Senseval2.key'
sense_embedding_file = 'senseval_sense_embedding.csv'

replace_target = re.compile("""<head.*?>.*</head>""")
replace_newline = re.compile("""\n""")
replace_dot = re.compile("\.")
replace_cite = re.compile("'")
replace_frac = re.compile("[\d]*frac[\d]+")
replace_num = re.compile("\s\d+\s")
rm_context_tag = re.compile('<.{0,1}context>')
rm_cit_tag = re.compile('\[[eb]quo\]')
rm_markup = re.compile('\[.+?\]')
rm_misc = re.compile("[\[\]\$`()%/,\.:;-]")

EMBEDDING_DIM = 100
total_words_wordnet = 206941 #  the total number of definitions in the dictionary WordNet 3.0

# ======================================================================================================================
# ======================================================================================================================

def clean_context(ctx_in):
    ctx = replace_target.sub(' <target> ', ctx_in)
    ctx = replace_newline.sub(' ', ctx)  # (' <eop> ', ctx)
    ctx = replace_dot.sub(' ', ctx)     # .sub(' <eos> ', ctx)
    ctx = replace_cite.sub(' ', ctx)    # .sub(' <cite> ', ctx)
    ctx = replace_frac.sub(' <frac> ', ctx)
    ctx = replace_num.sub(' <number> ', ctx)
    ctx = rm_cit_tag.sub(' ', ctx)
    ctx = rm_context_tag.sub('', ctx)
    ctx = rm_markup.sub('', ctx)
    ctx = rm_misc.sub('', ctx)
    return ctx

# ======================================================================================================================

def split_context(ctx):
    # word_list = re.split(', | +|\? |! |: |; ', ctx.lower())
    #print(ctx)
    word_list = [word for word in re.split(', | +|\? |! |: |; ', ctx.lower()) if word]
    #print(word_list)
    return word_list  #[stemmer.stem(word) for word in word_list]

# ======================================================================================================================

def one_hot_encode(length, target):
    y = np.zeros(length, dtype=np.float32)
    y[target] = 1.
    return y

# ======================================================================================================================

def load_train_data(se_2_or_3):
    if se_2_or_3 == 2:
        return load_senteval2_data(train_path2, True)
    elif se_2_or_3 == 3:
        return load_senteval3_data(train_path3, True)
    elif se_2_or_3 == 23:
        two = load_senteval2_data(train_path2, True)
        three = load_senteval3_data(train_path3, True)
        return two + three
    else:
        raise ValueError('2, 3 or 23. Provided: %d' % se_2_or_3)

# ======================================================================================================================

def load_test_data(se_2_or_3):
    if se_2_or_3 == 2:
        return load_senteval2_data(test_path2, False)
    elif se_2_or_3 == 3:
        return load_senteval3_data(test_path3, False)
    elif se_2_or_3 == 23:
        two = load_senteval2_data(test_path2, False)
        three = load_senteval3_data(test_path3, False)
        return two + three
    else:
        raise ValueError('2 or 3. Provided: %d' % se_2_or_3)

# ======================================================================================================================

def load_senteval3_data(path, is_training):
    return load_senteval2_data(path, is_training, False)

# ======================================================================================================================

def load_senteval2_data(path, is_training, dtd_validation=True):
    data = []
    parser = et.XMLParser(dtd_validation=dtd_validation)
    doc = et.parse(path, parser)
    instances = doc.findall('.//instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance:
            if child.tag == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':  # ignore
                    pass
                else:
                    answer = senseid
            elif child.tag == 'context':
                context = et.tostring(child)
                context = context.decode('utf-8')
            else:
                raise ValueError('unknown child tag to instance')

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            x = {
                'id': instance.get('id'),
                'docsrc': instance.get('docsrc'),
                'context': context,
                'target_sense': answer,  # todo support multiple answers?
                'target_word': instance.get('id').split('.')[0],
            }
            data.append(x)

    return data

# ======================================================================================================================

#Creates 3 dicts, 1 gives each word an id, 2 gives each sense an id, 3 relates number of senses to word id
def build_sense_ids(data):
    words = set()
    word_to_senses = {}
    for elem in data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word not in words:
            words.add(target_word)
            word_to_senses.update({target_word: [target_sense]})
        else:
            if target_sense not in word_to_senses[target_word]:
                word_to_senses[target_word].append(target_sense)
    
    words = list(words)
    target_word_to_id = dict(zip(words, range(len(words))))
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, len(words), n_senses_from_word_id

# ======================================================================================================================

def build_vocab(data):
    counter = collections.Counter()
    for elem in data:
        counter.update(split_context(elem['context']))
    min_freq = 1
    filtered = [item for item in counter.items() if item[1]>=min_freq]
    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

# ======================================================================================================================

#relates senses to context
def build_context(data, word_to_id):
    target_sense_to_context = {}
    for elem in data:
        target_sense_id = elem['id']
        #context = prepro(elem['context'])
        context = split_context(elem['context'])
        #context = sparse_matrix(context, word_to_id)
        if target_sense_id not in target_sense_to_context:
            #target_sense_to_context.update({target_sense:context})
            target_sense_to_context[target_sense_id] = []
        target_sense_to_context[target_sense_id].append(context)
    
    return target_sense_to_context

# ======================================================================================================================

def build_embedding(target_sense_to_context, embedding_matrix, word_num, EMBEDDING_DIM):
    res = {}
    wordvecs = load_glove(EMBEDDING_DIM)
    for target_sense_id, context_matrix in target_sense_to_context.items():
        embedded_sequences = np.zeros(EMBEDDING_DIM)
        n = 0
        for cont in context_matrix:
            for word in cont:
                n += 1
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                if word in wordvecs:
                    embedded_sequences += wordvecs[word]
                else:
                    embedded_sequences += np.random.normal(0.0, 0.1, EMBEDDING_DIM)                
        res[target_sense_id] = embedded_sequences/n
    return res

# ======================================================================================================================

def get_embedding(sense_embedding_file):
    sense_embeddings_ = None
    with open(sense_embedding_file, 'r', newline='') as f:
        reader = csv.reader(f)
        sense_embeddings_ = dict(reader)
    sense_embeddings__ = {}
    for key, value in sense_embeddings_.items():
        value = value.split()
        #if len(value) == 100 or len(value) == 102:
        #    print(value)
        vec = np.zeros(len(value)-1)
        for i in range(len(value)):
            if '[' in value[i]:                
                value[i] = value[i][1:]
            elif ']' in value[i]:
                value[i] = value[i][:-1]
            if value[i]:
                vec[i-1] = float(value[i])
        sense_embeddings__[key] = vec    
    return sense_embeddings__

# ======================================================================================================================

def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id,
                       target_sense_to_context_embedding, is_training=True, emb_size=100):
    
    n_senses_sorted_by_target_id = [n_senses_from_word_id[target_id] for target_id in range(len(n_senses_from_word_id))]
    starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
    tot_n_senses = sum(n_senses_from_word_id.values())
    all_data = []
    for instance in data:
        words = split_context(instance['context'])
        target_word = instance['target_word']
        ctx_ints = [word_to_id[word] for word in words if word in word_to_id]
        stop_idx = words.index('<target>')
        _instance = []
        xf = np.array(ctx_ints[:stop_idx])
        xb = np.array(ctx_ints[stop_idx+1:])[::-1]
        instance_id = instance['id']          
        target_id = target_word_to_id[target_word]
        
        _instance.append(xf)
        _instance.append(xb)
        _instance.append(instance_id)
        if is_training:                   
            target_sense = instance['target_sense']   
            if instance_id in target_sense_to_context_embedding:
                sense_embedding = target_sense_to_context_embedding[instance_id]
                senses = target_sense_to_id[target_id]
                #sense_id = senses[target_sense] if target_sense else -1
                _instance.append(sense_embedding)
        all_data.append(_instance[:])
    return all_data

# ======================================================================================================================

def group_by_target(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst[2]):
       res.update({key: list(group)})
    return res

# ======================================================================================================================

def group_by_target2(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst.target_id):
       res.update({key: list(group)})
    return res

# ======================================================================================================================

def split_grouped(data, frac, min=None):
    assert frac >= 0.
    assert frac < .5
    l = {}
    r = {}
    for target_id, instances in data.items():
        instances = [inst for inst in instances]
        random.shuffle(instances)   # optional
        n = len(instances)
        
        if frac == 0:
            l[target_id] = instances[:]
        else:        
            n_r = int(frac * n)
            if min and n_r < min:
                n_r = min
            n_l = n - n_r
    
            l[target_id] = instances[:n_l]
            r[target_id] = instances[-n_r:]

    return l, r if frac > 0 else l

# ======================================================================================================================

def get_data(_data, n_step_f, n_step_b):
    forward_data, backward_data, target_sense_ids, sense_embeddings = [], [], [], []
    for target_id, data in _data.items():
        for instance in data:
            xf, xb, target_sense_id, sense_embedding = instance[0], instance[1], instance[2], instance[3]
            
            n_to_use_f = min(n_step_f, len(xf))
            n_to_use_b = min(n_step_b, len(xb))
            xfs = np.zeros([n_step_f], dtype=np.int32)
            xbs = np.zeros([n_step_b], dtype=np.int32)            
            if n_to_use_f != 0:
                xfs[-n_to_use_f:] = xf[-n_to_use_f:]
            if n_to_use_b != 0:
                xbs[-n_to_use_b:] = xb[-n_to_use_b:]
            
            forward_data.append(xfs)
            backward_data.append(xbs)
            target_sense_ids.append(target_sense_id)
            sense_embeddings.append(sense_embedding)
    return (np.array(forward_data), np.array(backward_data), np.array(target_sense_ids), np.array(sense_embeddings))

# ======================================================================================================================

def get_data_test(_data, n_step_f, n_step_b):
    forward_data, backward_data, target_sense_ids = [], [], []
    for target_id, data in _data.items():
        for instance in data:
            xf, xb, target_sense_id = instance[0], instance[1], instance[2]
            n_to_use_f = min(n_step_f, len(xf))
            n_to_use_b = min(n_step_b, len(xb))
            xfs = np.zeros([n_step_f], dtype=np.int32)
            xbs = np.zeros([n_step_b], dtype=np.int32)
            if n_to_use_f != 0:
                xfs[-n_to_use_f:] = xf[-n_to_use_f:]
            if n_to_use_b != 0:
                xbs[-n_to_use_b:] = xb[-n_to_use_b:]
            forward_data.append(xfs)
            backward_data.append(xbs)
            target_sense_ids.append(target_sense_id)
    return (np.array(forward_data), np.array(backward_data), np.array(target_sense_ids))

# ======================================================================================================================
# ======================================================================================================================

if __name__ == '__main__':
    ''' Just gave an idea as to how the different methods are used and in what order '''
    data = load_senteval2_data(train_path2, True)
    test_data = load_senteval2_data(test_path2, False)
    word_to_id = build_vocab(data)
    target_word_to_id, target_sense_to_id, words_nums, n_senses_from_word_id = build_sense_ids(data)
    target_sense_to_context = build_context(data, word_to_id)
    embedding_matrix = fill_with_gloves(word_to_id, EMBEDDING_DIM)
    target_sense_to_context_embedding = build_embedding(target_sense_to_context, embedding_matrix, len(word_to_id), 100)
    ndata = convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_word_id,
                               target_sense_to_context_embedding, is_training=True)
    n_step_f = 40
    n_step_b = 40
    grouped_by_target = group_by_target(ndata)
    train_data, val_data = split_grouped(grouped_by_target, 0)
    train_forward_data, train_backward_data, train_target_sense_ids, train_sense_embedding = get_data(train_data,
                                                                                                      n_step_f, n_step_b)
    