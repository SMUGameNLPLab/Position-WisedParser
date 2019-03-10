__author__ = 'max'

import os.path
import random
import numpy as np
from .alphabet import Alphabet
from .logger import get_logger
from . import utils
import torch
from torch.autograd import Variable

# if USE KKMa
from konlpy.tag import Kkma

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_TYPE = "_<PAD>"
PAD_CHAR = "_PAD_CHAR"
ROOT = "_ROOT"
ROOT_POS = "_ROOT_POS"
ROOT_TYPE = "_<ROOT>"
ROOT_CHAR = "_ROOT_CHAR"
END = "_END"
END_POS = "_END_POS"
END_TYPE = "_<END>"
END_CHAR = "_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]


UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 1

NUM_SYMBOLIC_TAGS = 3

#_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 80, 130]
_buckets = [10, 15, 20, 25, 30, 35, 40, 50]

from .reader import CoNLLXReader


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=1000000, embedd_dict=None,
                     min_occurence=1, normalize_digits=True,kkma=False,end_to_end=False,syllable_postions=None):
    def expand_vocab():
        vocab_set = set(vocab_list)

        if kkma:
            tagger = Kkma()

        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r',encoding='utf-8-sig') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0 or line.startswith('#') or line=='\n':
                        continue
                    if end_to_end:

                        tokens = line.split('\t')
                        word = tokens[1]
                        word = utils.DIGIT_RE.sub("0",word) if normalize_digits else word

                        pre = word[0]
                        post = word[-1]

                        type = tokens[7]

                        type_alphabet.add(type)
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)
                        continue

                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    #for pos in tokens[4].split(" "):   #180815 XPOSTAG " " --> "+"
                    words = ""
                    if kkma:
                        pos_list = tagger.pos(tokens[1])
                        pos_str = ""
                        for word,pos in pos_list:
                            pos_alphabet.add(pos)
                            words = words + " "+word
                            pos_str = pos_str + "+"+pos
                        pos_str = pos_str[1:]
                        pos_alphabet.add(pos_str)
                        words = words[1:]
                        words = utils.DIGIT_RE.sub("0",words) if normalize_digits else words
                    else:
                        for pos in tokens[4].split("+"):
                            pos_alphabet.add(pos)
                        words = utils.DIGIT_RE.sub("0", tokens[2]) if normalize_digits else tokens[2]
                        pos_alphabet.add(tokens[4])

                    type = tokens[7]


                    type_alphabet.add(type)

                    for word in words.split(" "):
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True, singleton=True)
    pos_alphabet = Alphabet('pos', defualt_value=True, singleton=True)
    type_alphabet = Alphabet('type')

    if end_to_end:

        if syllable_postions[0]:
            syllable_begin_alphabet = Alphabet("pre_word", defualt_value=True, singleton=True)

        if syllable_postions[1]:
            syllable_begin2_alphabet = Alphabet("pre_bi_word", defualt_value=True, singleton=True)

        if syllable_postions[2]:
            syllable_last_alphabet = Alphabet("post_word",defualt_value=True,singleton=True)

        if syllable_postions[3]:
            syllable_last2_alphabet = Alphabet("post_bi_word",defualt_value=True,singleton=True)

    if kkma:
        tagger = Kkma()

    if True:
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = dict()

        syllable_begin_vocab = dict()
        syllable_begin2_vocab = dict()

        syllable_last_vocab = dict()
        syllable_last2_vocab = dict()

        with open(train_path, 'r',encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue
                tokens = line.split('\t')
                if end_to_end:
                    word = tokens[1]

                    word = utils.DIGIT_RE.sub("0",word) if normalize_digits else word

                    syllable_begin_word = utils.DIGIT_RE.sub("0",word[0]) if normalize_digits else word[0]
                    syllable_begin2_word = utils.DIGIT_RE.sub("0",word[:2]) if normalize_digits else word[:2]
                    syllable_last_word = utils.DIGIT_RE.sub("0",word[-1]) if normalize_digits else word[-1]
                    syllable_last2_word = utils.DIGIT_RE.sub("0",word[-2:]) if normalize_digits else word[-2:]

                    label = tokens[7]
                    type_alphabet.add(label)

                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

                    if syllable_postions[0]:
                        if syllable_begin_word in syllable_begin_vocab:
                            syllable_begin_vocab[syllable_begin_word] += 1
                        else:
                            syllable_begin_vocab[syllable_begin_word] = 1

                    if syllable_postions[1]:
                        if syllable_begin2_word in syllable_begin2_vocab:
                            syllable_begin2_vocab[syllable_begin2_word] += 1
                        else:
                            syllable_begin2_vocab[syllable_begin2_word] = 1

                    if syllable_postions[2]:
                        if syllable_last_word in syllable_last_vocab:
                            syllable_last_vocab[syllable_last_word] += 1
                        else:
                            syllable_last_vocab[syllable_last_word] = 1

                    if syllable_postions[3]:
                        if syllable_last2_word in syllable_last2_vocab:
                            syllable_last2_vocab[syllable_last2_word] += 1
                        else:
                            syllable_last2_vocab[syllable_last2_word]= 1

                    for char in tokens[1]:
                        char_alphabet.add(char)

                else:

                    for char in tokens[1]:
                        char_alphabet.add(char)

                #for pos in tokens[4].split(" "):   #180815 XPOSTAG " " --> "+"


                    if kkma:
                        pos_list = tagger.pos(tokens[1])
                        words = ""
                        pos_str = ""
                        for word,pos in pos_list:
                            pos_alphabet.add(pos)
                            words = words + " " + word
                            pos_str = pos_str + "+" + pos

                        words = words[1:]
                        pos_str = pos_str[1:]
                        pos_alphabet.add(pos_str)

                        words = utils.DIGIT_RE.sub("0",words) if normalize_digits else words
                    else:
                        for pos in tokens[4].split("+"):
                            pos_alphabet.add(pos)
                        words = utils.DIGIT_RE.sub("0", tokens[2]) if normalize_digits else tokens[2]
                        pos = tokens[4]

                    label = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(label)

                    for word in words.split(" "):
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
        # collect singletons

        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        if end_to_end:
            if syllable_postions[0]:
                syllable_begin_singletons = set([syllable_begin_word for syllable_begin_word, count
                                                 in syllable_begin_vocab.items() if count <= min_occurence])
            if syllable_postions[1]:
                syllable_begin2_singletons = set([syllable_begin2_word for syllable_begin2_word, count
                                                  in syllable_begin2_vocab.items() if count <= min_occurence])
            if syllable_postions[2]:
                syllable_last_singletons = set([syllable_last_word for syllable_last_word, count
                                                in syllable_last_vocab.items() if count <= min_occurence])
            if syllable_postions[3]:
                syllable_last2_singletons = set([syllable_last2_word for syllable_last2_word, count
                                                 in syllable_last2_vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if end_to_end:
            if syllable_postions[0]:
                syllable_begin_vocab_list = _START_VOCAB + sorted(syllable_begin_vocab, key=syllable_begin_vocab.get, reverse=True)
                logger.info("Total Syllable Begin Vocabulary Size: %d" % len(syllable_begin_vocab_list))
                logger.info("Total Syllable Begin SingleTon Size: %d" % len(syllable_begin_singletons))
                syllable_begin_vocab_list = [syllable_begin_word for syllable_begin_word
                                             in syllable_begin_vocab_list if syllable_begin_word
                                             in _START_VOCAB or syllable_begin_vocab[syllable_begin_word] > min_occurence]
                logger.info("Total Syllable Begin Vocabulary Size (w.o rare words): %d" % len(syllable_begin_vocab_list))
                if len(syllable_begin_vocab_list) > max_vocabulary_size:
                    syllable_begin_vocab_list = syllable_begin_vocab_list[:max_vocabulary_size]

            if syllable_postions[1]:
                syllable_begin2_vocab_list = _START_VOCAB + sorted(syllable_begin2_vocab,key=syllable_begin2_vocab.get, reverse=True)
                logger.info("Total Syllable Begin 2 Vocabulary Size: %d" % len(syllable_begin2_vocab_list))
                logger.info("Total Syllable Begin 2 SingleTon Size: %d" % len(syllable_begin2_singletons))
                syllable_begin2_vocab_list = [syllable_begin2_word for syllable_begin2_word
                                              in syllable_begin2_vocab_list if syllable_begin2_word
                                              in _START_VOCAB or syllable_begin2_vocab[syllable_begin2_word] > min_occurence]
                logger.info("Total Syllable Begin 2 Vocabulary Size (w.o rare words): %d" % len(syllable_begin2_vocab_list))

                if len(syllable_begin2_vocab_list) > max_vocabulary_size:
                    syllable_begin2_vocab_list = syllable_begin2_vocab_list[:max_vocabulary_size]


            if syllable_postions[2]:
                syllable_last_vocab_list = _START_VOCAB + sorted(syllable_last_vocab,key=syllable_last_vocab.get, reverse=True)
                logger.info("Total Syllable Last Vocabulary Size: %d" % len(syllable_last_vocab_list))
                logger.info("Total Syllable Last SingleTon Size: %d" % len(syllable_last_singletons))
                syllable_last_vocab_list = [syllable_last_word for syllable_last_word
                                            in syllable_last_vocab_list if syllable_last_word
                                            in _START_VOCAB or syllable_last_vocab[syllable_last_word] > min_occurence]

                logger.info("Total Syllable Last Vocabulary Size (w.o rare words): %d" % len(syllable_last_vocab_list))

                if len(syllable_last_vocab_list) > max_vocabulary_size:
                    syllable_last_vocab_list = syllable_last_vocab_list[:max_vocabulary_size]

            if syllable_postions[3]:
                syllable_last2_vocab_list = _START_VOCAB + sorted(syllable_last2_vocab,key=syllable_last2_vocab.get, reverse=True)
                logger.info("Total Syllable Last 2 Vocabulary Size: %d" % len(syllable_last2_vocab_list))
                logger.info("Total Syllable Last 2 SingleTon Size: %d" % len(syllable_last2_singletons))
                syllable_last2_vocab_list = [syllable_last2_word for syllable_last2_word
                                             in syllable_last2_vocab_list if syllable_last2_word
                                             in _START_VOCAB or syllable_last2_vocab[syllable_last2_word] > min_occurence]
                logger.info("Total Syllable Last 2 Vocabulary Size (w.o rare words): %d" % len(syllable_last2_vocab_list))

                if len(syllable_last2_vocab_list) > max_vocabulary_size:
                    syllable_last2_vocab_list = syllable_last2_vocab_list[:max_vocabulary_size]

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        if end_to_end:
            if syllable_postions[0]:
                for syllable_begin_word in syllable_begin_vocab_list:
                    syllable_begin_alphabet.add(syllable_begin_word)
                    if syllable_begin_word in syllable_begin_singletons:
                        syllable_begin_alphabet.add_singleton(syllable_begin_alphabet.get_index(syllable_begin_word))
            if syllable_postions[1]:
                for syllable_begin2_word in syllable_begin2_vocab_list:
                    syllable_begin2_alphabet.add(syllable_begin2_word)
                    if syllable_begin2_word in syllable_begin2_singletons:
                        syllable_begin2_alphabet.add_singleton(syllable_begin2_alphabet.get_index(syllable_begin2_word))

            if syllable_postions[2]:
                for syllable_last_word in syllable_last_vocab_list:
                    syllable_last_alphabet.add(syllable_last_word)
                    if syllable_last_word in syllable_last_singletons:
                        syllable_last_alphabet.add_singleton(syllable_last_alphabet.get_index(syllable_last_word))
            if syllable_postions[3]:
                for syllable_last2_word in syllable_last2_vocab_list:
                    syllable_last2_alphabet.add(syllable_last2_word)
                    if syllable_last2_word in syllable_last2_singletons:
                        syllable_last2_alphabet.add_singleton(syllable_last2_alphabet.get_index(syllable_last2_word))

        word_alphabet.save(alphabet_directory)
        if end_to_end:
            if syllable_postions[0]:
                syllable_begin_alphabet.save(alphabet_directory)

            if syllable_postions[1]:
                syllable_begin2_alphabet.save(alphabet_directory)

            if syllable_postions[2]:
                syllable_last_alphabet.save(alphabet_directory)

            if syllable_postions[3]:
                syllable_last2_alphabet.save(alphabet_directory)

        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        if end_to_end:
            if syllable_postions[0]:
                syllable_begin_alphabet.load(alphabet_directory)
            if syllable_postions[1]:
                syllable_begin2_alphabet.load(alphebet_directory)
            if syllable_postions[2]:
                syllable_last_alphabet.load(alphabet_directory)
            if syllable_postions[3]:
                syllable_last2_alphabet.load(alphabet_directory)

        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    if end_to_end:
        if syllable_postions[0]:
            syllable_begin_alphabet.close()
        if syllable_postions[1]:
            syllable_begin2_alphabet.close()
        if syllable_postions[2]:
            syllable_last_alphabet.close()
        if syllable_postions[3]:
            syllable_last2_alphabet.close()


    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))

    if end_to_end:
        logger.info("Prefix Word Alphabet Size (Singleton): %d (%d)" % (syllable_begin_alphabet.size(), syllable_begin_alphabet.singleton_size()))
        logger.info("Postfix Word Alphabet Size (Singleton): %d (%d)" % (syllable_last_alphabet.size(), syllable_last_alphabet.singleton_size()))
        logger.info("Postfix BI Word Alphabet Size (Singleton): %d (%d)" % (syllable_last2_alphabet.size(), syllable_last2_alphabet.singleton_size()))

    logger.info("Character Alphabet Size (Singleton): %d (%d)" % (char_alphabet.size(), char_alphabet.singleton_size()))
    logger.info("POS Alphabet Size (Singleton): %d (%d)" % (pos_alphabet.size(), pos_alphabet.singleton_size()))
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())

    if end_to_end:
        syllable_alphabets = [None for _ in range(4)]
        if syllable_postions[0]:
            syllable_alphabets[0] = syllable_begin_alphabet

        if syllable_postions[1]:
            syllable_alphabets[1] = syllable_begin2_alphabet

        if syllable_postions[2]:
            syllable_alphabets[2] = syllable_last_alphabet
        if syllable_postions[3]:
            syllable_alphabets[3] = syllable_last2_alphabet

        return (word_alphabet, syllable_alphabets), char_alphabet, syllable_last_alphabet, type_alphabet

    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet

def load_alphabets(alphabet_directory):
    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True, singleton=True)
    pos_alphabet = Alphabet('pos', defualt_value=True, singleton=True)
    type_alphabet = Alphabet('type')

    if os.path.isdir(alphabet_directory):
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)
    else:
        print("Input the alphabets dictionary file.")
        exit()

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()

    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size (Singleton): %d (%d)" % (char_alphabet.size(), char_alphabet.singleton_size()))
    logger.info("POS Alphabet Size (Singleton): %d (%d)" % (pos_alphabet.size(), pos_alphabet.singleton_size()))
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False,kkma=False,end_to_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,kkma=kkma,end_to_end=end_to_end)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, sent_id=counter)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, sent_id=counter)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[b, :inst_size] = tids
        tid_inputs[b, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, hid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], hid_inputs[excerpt], \
                  tid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
                          normalize_digits=True, symbolic_root=False, symbolic_end=False,
                          use_gpu=False, volatile=False):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        heads = Variable(torch.from_numpy(hid_inputs), volatile=volatile)
        types = Variable(torch.from_numpy(tid_inputs), volatile=volatile)
        masks = Variable(torch.from_numpy(masks), volatile=volatile)
        single = Variable(torch.from_numpy(single), volatile=volatile)
        lengths = torch.from_numpy(lengths)
        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks = masks.cuda()
            single = single.cuda()
            lengths = lengths.cuda()

        data_variable.append((words, chars, pos, heads, types, masks, single, lengths))

    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0.):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
        noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, heads, types, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                  masks[excerpt], lengths[excerpt]
