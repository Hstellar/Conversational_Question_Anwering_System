import re
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random
from allennlp.modules.elmo import batch_to_ids
from general_utils import flatten_json, free_text_to_span, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, find_answer_span, feature_gen, token2id

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 15 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')
parser.add_argument('--train_file', default='CoQA/train.json',
                    help='Add the input train data file')
parser.add_argument('--dev_file', default='CoQA/dev.json',
                    help='Add the input valid data file')
args = parser.parse_args()
trn_file = args.train_file
dev_file = args.dev_file
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en_core_web_sm', disable=['parser'])
random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#=======================================================================
#=================== Work on training and dev data =====================
#=======================================================================

def proc_train(ith, article):
    rows = []
    context = article['story']

    for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
        gold_answer = answers['input_text']
        span_answer = answers['span_text']

        answer, char_i, char_j = free_text_to_span(gold_answer, span_answer)
        answer_choice = 0 if answer == '__NA__' else\
                        1 if answer == '__YES__' else\
                        2 if answer == '__NO__' else\
                        3 # Not a yes/no question

        if answer_choice == 3:
            answer_start = answers['span_start'] + char_i
            answer_end = answers['span_start'] + char_j
        else:
            answer_start, answer_end = -1, -1

        rationale = answers['span_text']
        rationale_start = answers['span_start']
        rationale_end = answers['span_end']

        q_text = question['input_text']
        if j > 0:
            q_text = article['answers'][j-1]['input_text'] + " // " + q_text

        rows.append((ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
    return rows, context

def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in glove_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in glove_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    vocab.insert(2, "<S>")
    vocab.insert(3, "</S>")
    return vocab

def build_dev_vocab(questions, contexts,tr_vocab): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

def Data_preprocessing(file,datatype,vocab = None,vocab_tag = None,vocab_ent = None):
    data, data_context = flatten_json(file, proc_train)
    data = pd.DataFrame(data, columns=['context_idx', 'question', 'answer', 'answer_start', 'answer_end', 'rationale', 'rationale_start', 'rationale_end', 'answer_choice'])
    log.info('{} json data flattened.'.format(datatype))
    print(data)
    C_iter = (pre_proc(c) for c in data_context)
    Q_iter = (pre_proc(q) for q in data.question)
    C_docs = [doc for doc in nlp.pipe(C_iter, batch_size=64, n_process=args.threads)]
    Q_docs = [doc for doc in nlp.pipe(Q_iter, batch_size=64, n_process=args.threads)]
    print(C_docs)
    # tokens
    C_tokens = [[normalize_text(w.text) for w in doc] for doc in C_docs]
    Q_tokens = [[normalize_text(w.text) for w in doc] for doc in Q_docs]
    C_unnorm_tokens = [[w.text for w in doc] for doc in C_docs]
    log.info('All tokens for {} are obtained.'.format(datatype))
    data_context_span = [get_context_span(a, b) for a, b in zip(data_context, C_unnorm_tokens)]
    ans_st_token_ls, ans_end_token_ls = [], []
    for ans_st, ans_end, idx in zip(data.answer_start, data.answer_end, data.context_idx):
        ans_st_token, ans_end_token = find_answer_span(data_context_span[idx], ans_st, ans_end)
        ans_st_token_ls.append(ans_st_token)
        ans_end_token_ls.append(ans_end_token)
    ration_st_token_ls, ration_end_token_ls = [], []
    for ration_st, ration_end, idx in zip(data.rationale_start, data.rationale_end, data.context_idx):
        ration_st_token, ration_end_token = find_answer_span(data_context_span[idx], ration_st, ration_end)
        ration_st_token_ls.append(ration_st_token)
        ration_end_token_ls.append(ration_end_token)
    data['answer_start_token'], data['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
    data['rationale_start_token'], data['rationale_end_token'] = ration_st_token_ls, ration_end_token_ls
    initial_len = len(data)
    data.dropna(inplace=True) # modify self DataFrame
    log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(data), initial_len))
    log.info('answer span for {} is generated.'.format(datatype))

    # features
    C_tags, C_ents, C_features = feature_gen(C_docs, data.context_idx, Q_docs, args.no_match)
    log.info('features for training is generated: {}, {}, {}'.format(len(C_tags), len(C_ents), len(C_features)))
    # vocab
    if(datatype == "train"):
        vocab = build_train_vocab(Q_tokens, C_tokens)
    else:
        vocab = build_dev_vocab(Q_tokens, C_tokens,vocab)

    C_ids = token2id(C_tokens, vocab, unk_id=1)
    Q_ids = token2id(Q_tokens, vocab, unk_id=1)
    Q_tokens = [["<S>"] + doc + ["</S>"] for doc in Q_tokens]
    Q_ids = [[2] + qsent + [3] for qsent in Q_ids]
    print(Q_ids[:10])
    # tags
    if(datatype == "train"):
        vocab_tag = [''] + list(nlp.tagger.labels)
    C_tag_ids = token2id(C_tags, vocab_tag)
    # entities
    if(datatype == "train"):
        vocab_ent = list(set([ent for sent in C_ents for ent in sent]))
    vocab_ent = list(set([ent for sent in C_ents for ent in sent]))
    C_ent_ids = token2id(C_ents, vocab_ent, unk_id=0)

    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
    log.info('vocabulary for training is built.')
    data_embedding = build_embedding(wv_file, vocab, wv_dim)
    log.info('got embedding matrix for {}.'.format(datatype))

    meta = {
    'vocab': vocab,
    'embedding': data_embedding.tolist()
    }
    path = 'CoQA/{}_meta.msgpack'.format(datatype)
    with open(path, 'wb') as f:
        msgpack.dump(meta, f)
    del meta
    del data_embedding
    prev_CID, first_question = -1, []
    for i, CID in enumerate(data.context_idx):
        if not (CID == prev_CID):
            first_question.append(i)
        prev_CID = CID
    result = {
        'question_ids': Q_ids,
        'context_ids': C_ids,
        'context_features': C_features, # exact match, tf
        'context_tags': C_tag_ids, # POS tagging
        'context_ents': C_ent_ids, # Entity recognition
        'context': data_context,
        'context_span': data_context_span,
        '1st_question': first_question,
        'question_CID': data.context_idx.tolist(),
        'question': data.question.tolist(),
        'answer': data.answer.tolist(),
        'answer_start': data.answer_start_token.tolist(),
        'answer_end': data.answer_end_token.tolist(),
        'rationale_start': data.rationale_start_token.tolist(),
        'rationale_end': data.rationale_end_token.tolist(),
        'answer_choice': data.answer_choice.tolist(),
        'context_tokenized': C_tokens,
        'question_tokenized': Q_tokens
    }
    path = 'CoQA/{}_data.msgpack'.format(datatype)
    with open(path, 'wb') as f:
        msgpack.dump(result, f)
    del result
    log.info('saved training to disk.')
    if(datatype == "train"):
        return vocab,vocab_tag,vocab_ent
    
vocab,vocab_tag,vocab_ent = Data_preprocessing(trn_file,"train")
Data_preprocessing(dev_file,"dev",vocab,vocab_tag,vocab_ent)
