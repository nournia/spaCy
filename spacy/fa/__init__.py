# coding: utf8
from __future__ import unicode_literals
from os import path

from .. import orth
from ..vocab import Vocab
from ..tokenizer import Tokenizer


from ..en.pos import POS_TAGS
from ..en.attrs import get_flags

def get_lex_props(string, oov_prob=-30, is_oov=False):
    return {
        'flags': get_flags(string, is_oov=is_oov),
        'length': len(string),
        'orth': string,
        'lower': string.lower(),
        'norm': string,
        'shape': orth.word_shape(string),
        'prefix': string[0],
        'suffix': string[-3:],
        'cluster': 0,
        'prob': oov_prob,
        'sentiment': 0
    }

LOCAL_DATA_DIR = path.join(path.dirname(__file__), 'data')


class Persian(object):
    """The Persian NLP pipeline.
    """

    def __init__(self, data_dir=LOCAL_DATA_DIR, Tokenizer=Tokenizer.from_dir):

        self.data_dir = data_dir

        self.vocab = Vocab(data_dir=path.join(data_dir, 'vocab') if data_dir else None,
                           get_lex_props=get_lex_props, pos_tags=POS_TAGS,
                           load_vectors=None, oov_prob=None)

        self.tokenizer = Tokenizer(self.vocab, path.join(data_dir, 'tokenizer'))

    def __call__(self, text):
        """Apply the pipeline to some text. The text can span multiple sentences,
        and can contain arbtrary whitespace. Alignment into the original string
        is preserved.

        Args:
            text (unicode): The text to be processed.

        Returns:
            tokens (spacy.tokens.Doc):

        >>> from spacy.fa import Persian
        >>> nlp = Persian()
        >>> tokens = nlp('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')
        >>> tokens[0].orth_
        'ما'
        """
        tokens = self.tokenizer(text)
        return tokens
