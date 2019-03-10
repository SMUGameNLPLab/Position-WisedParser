__author__ = 'max'

import numpy as np
import torch
from torch.autograd import Variable
from .conllx_data import _buckets, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, UNK_ID
from .conllx_data import NUM_SYMBOLIC_TAGS
from .conllx_data import create_alphabets, load_alphabets
from . import utils
from .reader import CoNLLXReader


def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


def _generate_stack_inputs(heads, types, prior_order):
    if prior_order == 'deep_first':
        child_ids = _obtain_child_index_for_depth(heads, True)
    elif prior_order == 'shallow_first':
        child_ids = _obtain_child_index_for_depth(heads, False)
    elif prior_order == 'left2right':
        child_ids = _obtain_child_index_for_left2right(heads)
    elif prior_order == 'inside_out':
        child_ids = _obtain_child_index_for_inside_out(heads)
    else:
        raise ValueError('Unknown prior order: %s' % prior_order)

    stacked_heads = []
    children = []
    siblings = []
    stacked_types = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    stack = [0]
    position = 1
    while len(stack) > 0:
        head = stack[-1]
        stacked_heads.append(head)
        siblings.append(sibs[head])
        child_id = child_ids[head]
        skip_connect.append(prev[head])
        prev[head] = position
        if len(child_id) == 0:
            children.append(head)
            sibs[head] = 0
            stacked_types.append(PAD_ID_TAG)
            stack.pop()
        else:
            child = child_id.pop(0)
            children.append(child)
            sibs[head] = child
            stack.append(child)
            stacked_types.append(types[child])
        position += 1

    return stacked_heads, children, siblings, stacked_types, skip_connect


def read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                      max_size=None, normalize_digits=True, prior_order='deep_first',kkma=False,
                      end_to_end=False, syllable_positions=None):
    data = [[] for _ in _buckets]
    max_lemma_length = [0 for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                          kkma, end_to_end,syllable_positions)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size <= bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)

                if end_to_end:
                    data[bucket_id].append([sent.word_ids,sent.pre_word_ids,sent.post_word_ids,sent.post_bi_word_ids,sent.pre_bi_word_ids,
                                            sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect, sent.sentence])
                else:
                    data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect, sent.sentence])

                char_lengths = []
                for eojul in sent.char_seqs:
                    char_lengths.append(len(eojul))

                max_len = max(char_lengths)
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len

                max_len = max(list(map(lambda x: len(x),inst.postags)))
                if max_lemma_length[bucket_id] < max_len:
                    max_lemma_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_lemma_length, max_char_length


def read_stacked_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                  max_size=None, normalize_digits=True, prior_order='deep_first', use_gpu=False,kkma=False,
                                  end_to_end=False,syllable_positions=None):

    data, max_lemma_length, max_char_length = read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                                max_size=max_size, normalize_digits=normalize_digits, prior_order=prior_order,
                                                                kkma=kkma,end_to_end=end_to_end,syllable_positions=syllable_positions)
    if end_to_end:
        word_alphabet, (syllable_begin_alphabet, syllable_begin2_alphabet,\
            syllable_last_alphabet, syllable_last2_alphabet) = word_alphabet

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        lemma_length = min(utils.MAX_EOJUL_LENGTH, max_lemma_length[bucket_id] + utils.NUM_EOJUL_PAD)
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        if end_to_end:
            wid_inputs = np.empty([bucket_size,bucket_length],dtype=np.int64)
            syllable_begin_wid_inputs = np.empty([bucket_size,bucket_length],dtype=np.int64)
            syllable_begin2_wid_inputs = np.empty([bucket_size,bucket_length],dtype=np.int64)
            syllable_last_wid_inputs = np.empty([bucket_size,bucket_length],dtype=np.int64)
            syllable_last2_wid_inputs = np.empty([bucket_size,bucket_length],dtype=np.int64)

            cid_inputs = np.empty([bucket_size,bucket_length,char_length],dtype=np.int64)
        else:
            wid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)

            cid_inputs = np.empty([bucket_size, bucket_length, lemma_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length, lemma_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        if end_to_end:
            single = np.zeros([bucket_size,bucket_length],dtype=np.int64)

            syllable_begin_single = np.zeros([bucket_size,bucket_length],dtype=np.int64)
            syllable_begin2_single = np.zeros([bucket_size,bucket_length],dtype=np.int64)

            syllable_last_single = np.zeros([bucket_size,bucket_length],dtype=np.int64)
            syllable_last2_single = np.zeros([bucket_size,bucket_length],dtype=np.int64)
        else:
            single = np.zeros([bucket_size, bucket_length, lemma_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        sentences = []

        for i, inst in enumerate(data[bucket_id]):
            if end_to_end:
                wids,syllable_begin_wids,syllable_begin2_wids,syllable_last_wids,syllable_last2_wids,cid_seqs\
                    ,pids,hids,tids,stack_hids,chids,ssids,stack_tids,skip_ids,sentence = inst
            else:
                wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids, sentence = inst
            inst_size = len(wids)
            lengths_e[i] = inst_size
            # word ids
            if end_to_end:
                wid_inputs[i,:inst_size] = wids
                wid_inputs[i,inst_size:] = PAD_ID_WORD
                syllable_begin_wid_inputs[i,:inst_size] = syllable_begin_wids
                syllable_begin_wid_inputs[i,inst_size:] = PAD_ID_WORD
                syllable_last_wid_inputs[i,:inst_size] = syllable_last_wids
                syllable_last_wid_inputs[i,inst_size:] = PAD_ID_WORD
                syllable_last2_wid_inputs[i,:inst_size] = syllable_last2_wids
                syllable_last2_wid_inputs[i,inst_size:] = PAD_ID_WORD
                syllable_begin2_wid_inputs[i,:inst_size] = syllable_begin2_wids
                syllable_begin2_wid_inputs[i,inst_size:] = PAD_ID_WORD
                for c,cids in enumerate(cid_seqs):
                    c_len = len(cids)
                    cid_inputs[i,c,:c_len] = cids
                    cid_inputs[i,c,c_len:] = PAD_ID_CHAR
            else:
                for w, w_ids in enumerate(wids):
                    wid_inputs[i, w, :len(w_ids)] = w_ids
                    wid_inputs[i, w, len(w_ids):] = PAD_ID_WORD
                wid_inputs[i, inst_size:, :] = PAD_ID_WORD

                # char ids
                for c, cids in enumerate(cid_seqs):
                    for l, lids in enumerate(cids):
                        cid_inputs[i, c, l, :len(lids)] = lids
                        cid_inputs[i, c, l, len(lids):] = PAD_ID_CHAR
                    cid_inputs[i, c, len(cids):, :] = PAD_ID_CHAR
                cid_inputs[i, inst_size:, :, :] = PAD_ID_CHAR

                # pos ids
                for p, p_ids in enumerate(pids):
                    pid_inputs[i, p, :len(p_ids)] = p_ids
                    pid_inputs[i, p, len(p_ids):] = PAD_ID_TAG
                pid_inputs[i, inst_size:, :] = PAD_ID_TAG

            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks_e
            masks_e[i, :inst_size] = 1.0
            if end_to_end:
                for j,lids in enumerate(wids):
                    if word_alphabet.is_singleton(lids):
                        single[i,j]=1
                for j,lids in enumerate(syllable_begin_wids):
                    if syllable_begin_alphabet.is_singleton(lids):
                        syllable_begin_single[i,j]=1

                for j, lids in enumerate(syllable_begin2_wids):
                    if syllable_begin2_alphabet.is_singleton(lids):
                        syllable_begin2_single[i, j] = 1

                for j, lids in enumerate(syllable_last_wids):
                    if syllable_last_alphabet.is_singleton(lids):
                        syllable_last_single[i, j] = 1
                for j, lids in enumerate(syllable_last2_wids):
                    if syllable_last2_alphabet.is_singleton(lids):
                        syllable_last2_single[i,j]=1

            else:
                for j, lids in enumerate(wids):
                    for k, wid in enumerate(lids):
                        if word_alphabet.is_singleton(wid):
                            single[i, j, k] = 1

            inst_size_decoder = 2 * inst_size - 1
            lengths_d[i] = inst_size_decoder
            # stacked heads
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # children
            chid_inputs[i, :inst_size_decoder] = chids
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # siblings
            ssid_inputs[i, :inst_size_decoder] = ssids
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # stacked types
            stack_tid_inputs[i, :inst_size_decoder] = stack_tids
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # skip connects
            skip_connect_inputs[i, :inst_size_decoder] = skip_ids
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
            # masks_d
            masks_d[i, :inst_size_decoder] = 1.0
            sentences.append(sentence)

        words = torch.from_numpy(wid_inputs)
        if end_to_end:
            syllable_begins = torch.from_numpy(syllable_begin_wid_inputs)
            syllable_begin2s = torch.from_numpy(syllable_begin2_wid_inputs)
            syllable_lasts = torch.from_numpy(syllable_last_wid_inputs)
            syllable_last2s = torch.from_numpy(syllable_last2_wid_inputs)

        chars = torch.from_numpy(cid_inputs)
        pos = torch.from_numpy(pid_inputs)
        heads = torch.from_numpy(hid_inputs)
        types = torch.from_numpy(tid_inputs)
        masks_e = torch.from_numpy(masks_e)
        single = torch.from_numpy(single)
        if end_to_end:
            syllable_begin_single = torch.from_numpy(syllable_begin_single)
            syllable_begin2_single = torch.from_numpy(syllable_begin2_single)

            syllable_last_single = torch.from_numpy(syllable_last_single)
            syllable_last2_single = torch.from_numpy(syllable_last2_single)
        lengths_e = torch.from_numpy(lengths_e)

        stacked_heads = torch.from_numpy(stack_hid_inputs)
        children = torch.from_numpy(chid_inputs)
        siblings = torch.from_numpy(ssid_inputs)
        stacked_types = torch.from_numpy(stack_tid_inputs)
        skip_connect = torch.from_numpy(skip_connect_inputs)
        masks_d = torch.from_numpy(masks_d)
        lengths_d = torch.from_numpy(lengths_d)

        if False:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            heads = heads.cuda()
            types = types.cuda()
            masks_e = masks_e.cuda()
            single = single.cuda()
            lengths_e = lengths_e.cuda()
            stacked_heads = stacked_heads.cuda()
            children = children.cuda()
            siblings = siblings.cuda()
            stacked_types = stacked_types.cuda()
            skip_connect = skip_connect.cuda()
            masks_d = masks_d.cuda()
            lengths_d = lengths_d.cuda()
        if end_to_end:
            data_variable.append(((words,syllable_begins,syllable_begin2s,syllable_lasts,syllable_last2s, chars, pos, heads, types,
                                   masks_e, single,syllable_begin_single,syllable_begin2_single,
                                   syllable_last_single,syllable_last2_single,
                                   lengths_e),
                              (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d),
                              sentences))

        else:
            data_variable.append(((words, chars, pos, heads, types, masks_e, single, lengths_e),
                              (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d),
                              sentences ))
    return data_variable, bucket_sizes


def get_batch_stacked_variable(data, batch_size, unk_replace=0., use_gpu=False,end_to_end = False):
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

    data_encoder, data_decoder, _ = data_variable[bucket_id]
    if end_to_end:
        words,syllable_begins,syllable_begin2s,syllable_lasts,syllable_last2s, chars, pos, heads, types, masks_e, single, syllable_begin_single, syllable_last_single,syllable_last2_single,syllable_begin2_single, lengths_e = data_encoder
    else:
        words, chars, pos, heads, types, masks_e, single, lengths_e = data_encoder
    stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    # if words.is_cuda:
    #     index = index.cuda()
    if not end_to_end:
        lemma_length = words.size(2)

    words = words[index]

    if end_to_end:
        syllable_begins = syllable_begins[index]
        syllable_begin2s = syllable_begin2s[index]
        syllable_lasts = syllable_lasts[index]
        syllable_last2s = syllable_last2s[index]
    if unk_replace:
        if end_to_end:
            ones = torch.LongTensor(single.data.new(batch_size,bucket_length).fill_(1))
            noise = torch.LongTensor(masks_e.data.new(batch_size,bucket_length).bernoulli_(unk_replace).long())

            syllable_begin_one = torch.LongTensor(syllable_begin_single.data.new(batch_size,bucket_length).fill_(1))
            syllable_begin_noise = torch.LongTensor(masks_e.data.new(batch_size,bucket_length).bernoulli_(unk_replace).long())

            syllable_begin2_one = torch.LongTensor(syllable_begin2_single.data.new(batch_size,bucket_length).fill_(1))
            syllable_begin2_noise = torch.LongTensor(masks_e.data.new(batch_size,bucket_length).bernoulli_(unk_replace).long())

            syllable_last_one = torch.LongTensor(syllable_last_single.data.new(batch_size,bucket_length).fill_(1))
            syllable_last_noise = torch.LongTensor(masks_e.data.new(batch_size,bucket_length).bernoulli_(unk_replace).long())

            syllable_last2_one = torch.LongTensor(syllable_last2_single.data.new(batch_size,bucket_length).fill_(1))
            syllable_last2_noise = torch.LongTensor(masks_e.data.new(batch_size,bucket_length).bernoulli_(unk_replace).long())

            words = words * (ones - single[index] * noise)

            syllable_begins = syllable_begins * (syllable_begin_one - syllable_begin_single[index] * syllable_begin_noise)
            syllable_begin2s = syllable_begin2s * (syllable_begin2_one - syllable_begin2_single[index] * syllable_begin2_noise)

            syllable_lasts = syllable_lasts * (syllable_last_one - syllable_last_single[index] * syllable_last_noise)
            syllable_last2s = syllable_last2s * (syllable_last2_one - syllable_last2_single[index] * syllable_last2_noise)
        else:
            ones = torch.LongTensor(single.data.new(batch_size, bucket_length, lemma_length).fill_(1))
            noise = torch.LongTensor(masks_e.data.new(batch_size, bucket_length, lemma_length).bernoulli_(unk_replace).long())
            words = words * (ones - single[index] * noise)

    if use_gpu:
        words = words.cuda()
        if end_to_end:
            syllable_begins = syllable_begins.cuda()
            syllable_begin2s = syllable_begin2s.cuda()

            syllable_lasts = syllable_lasts.cuda()
            syllable_last2s = syllable_last2s.cuda()
        chars = chars.cuda()
        pos = pos.cuda()
        heads = heads.cuda()
        types = types.cuda()
        masks_e = masks_e.cuda()
        lengths_e = lengths_e.cuda()
        stacked_heads = stacked_heads.cuda()
        children = children.cuda()
        siblings = siblings.cuda()
        stacked_types = stacked_types.cuda()
        skip_connect = skip_connect.cuda()
        masks_d = masks_d.cuda()
        lengths_d = lengths_d.cuda()
    if end_to_end:
        return ((words,syllable_begins,syllable_begin2s,syllable_lasts,syllable_last2s), chars[index], pos[index], heads[index], types[index], masks_e[index], lengths_e[index]), \
           (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], masks_d[index], lengths_d[index])
    else:
        return (words, chars[index], pos[index], heads[index], types[index], masks_e[index],
                lengths_e[index]), \
               (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index],
                masks_d[index], lengths_d[index])


def iterate_batch_stacked_variable(data, batch_size, unk_replace=0., shuffle=False, use_gpu=False,end_to_end=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        data_encoder, data_decoder, data_sentences = data_variable[bucket_id]
        if end_to_end:
            words,syllable_begin,syllable_begin2,syllable_last,syllable_last2,\
                chars, pos, heads, types, masks_e, \
                single,syllable_begin_single,syllable_begin2_single,syllable_last_single,syllable_last2_single, \
                lengths_e = data_encoder
        else:
            words, chars, pos, heads, types, masks_e, single, lengths_e = data_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_decoder
        if not end_to_end:
            lemma_length = words.size(2)
        if unk_replace:
            if end_to_end:
                ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
                noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
                syllable_begin_ones = Variable(syllable_begin_single.data.new(bucket_size, bucket_length).fill_(1))
                syllable_begin_noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())

                syllable_begin2_ones = Variable(syllable_begin2_single.data.new(bucket_size, bucket_length).fill_(1))
                syllable_begin2_noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())

                syllable_last_ones = Variable(syllable_last_single.data.new(bucket_size, bucket_length).fill_(1))
                syllable_last_noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())

                syllable_last2_ones = Variable(syllable_last2_single.data.new(bucket_size, bucket_length).fill_(1))
                syllable_last2_noise = Variable(masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())

                words = words * (ones - single * noise)

                syllable_begin = syllable_begin * (syllable_begin_ones - syllable_begin_single * syllable_begin_noise)
                syllable_last = syllable_last * (syllable_last_ones - syllable_last_single * syllable_last_noise)
                syllable_last2 = syllable_last2 * (syllable_last2_ones - syllable_last2_single * syllable_last2_noise)
                syllable_begin2 = syllable_begin2 * (syllable_begin2_ones - syllable_begin2_single * syllable_begin2_noise)

            else:
                ones = Variable(single.data.new(bucket_size, bucket_length, lemma_length).fill_(1))
                noise = Variable(masks_e.data.new(bucket_size, bucket_length, lemma_length).bernoulli_(unk_replace).long())
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
            if use_gpu:
                if end_to_end:
                    yield ((words[excerpt].cuda(),syllable_begin[excerpt].cuda(),syllable_begin2[excerpt].cuda(),syllable_last[excerpt].cuda(),syllable_last2[excerpt].cuda()), chars[excerpt].cuda(), pos[excerpt].cuda(), heads[excerpt].cuda(), types[excerpt].cuda(), masks_e[excerpt].cuda(), lengths_e[excerpt].cuda()), \
                      (stacked_heads[excerpt].cuda(), children[excerpt].cuda(), siblings[excerpt].cuda(), stacked_types[excerpt].cuda(), skip_connect[excerpt].cuda(), masks_d[excerpt].cuda(), lengths_d[excerpt].cuda()), \
                      data_sentences[excerpt]

                else:
                    yield (words[excerpt].cuda(), chars[excerpt].cuda(), pos[excerpt].cuda(), heads[excerpt].cuda(), types[excerpt].cuda(), masks_e[excerpt].cuda(), lengths_e[excerpt].cuda()), \
                      (stacked_heads[excerpt].cuda(), children[excerpt].cuda(), siblings[excerpt].cuda(), stacked_types[excerpt].cuda(), skip_connect[excerpt].cuda(), masks_d[excerpt].cuda(), lengths_d[excerpt].cuda()), \
                      data_sentences[excerpt]
            else:
                if end_to_end:
                    yield ((words[excerpt],syllable_begin[excerpt],syllable_begin2[excerpt],syllable_last[excerpt],syllable_last2[excerpt]), chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], masks_e[excerpt], lengths_e[excerpt]), \
                      (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], masks_d[excerpt], lengths_d[excerpt]), \
                      data_sentences[excerpt]

                else:
                    yield (words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], masks_e[excerpt], lengths_e[excerpt]), \
                      (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], masks_d[excerpt], lengths_d[excerpt]), \
                      data_sentences[excerpt]
