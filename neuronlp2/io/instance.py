__author__ = 'max'


class Sentence(object):
    def __init__(self, words, word_ids, pre_words=None, pre_word_ids=None, post_words=None, post_word_ids=None,post_bi_words=None,post_bi_word_ids=None, pre_bi_words=None,pre_bi_word_ids=None,
                 char_seqs=None, char_id_seqs=None, sent_id=None, sentence=None):
        self.words = words
        self.word_ids = word_ids
        self.pre_words = pre_words
        self.pre_word_ids = pre_word_ids
        self.post_words = post_words
        self.post_word_ids = post_word_ids
        self.post_bi_words = post_bi_words
        self.post_bi_word_ids = post_bi_word_ids
        self.pre_bi_words = pre_bi_words
        self.pre_bi_word_ids = pre_bi_word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs
        self.sent_id = sent_id
        self.sentence = sentence

    def length(self):
        return len(self.words)

    def get_sent_id(self):
        return self.sent_id

    def get_sentence(self):
        return self.sentence


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return self.sentence.length()


class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()