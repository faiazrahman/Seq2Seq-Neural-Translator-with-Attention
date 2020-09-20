from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

from language import Language

PATH_TO_DATA = '../data/'

def normalize_string(s):
    """
    Normalizes a string, which includes converting it to ASCII
    """

    def unicode_to_ascii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
    )

    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_languages(language1, language2, reverse=False):
    """
    Reads the data for translation FROM language1 TO language2
    by default; reads FROM language2 TO language1 if reverse=True

    Params
        language1   (str) 3-letter abbreviation of language
        language2   (str) 3-letter abbreviation of language
        reverse     Specifies language2 -> language1 translation

    Returns
        input_language  (Language)
        output_language (Language)
        pairs           (list)
    """
    
    lines = open(PATH_TO_DATA + '%s-%s.txt' % (language1, language2),
                 encoding='utf-8').read().strip().split('\n')
    
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_language = Language(language2)
        output_language = Language(language1)
    else:
        input_language = Language(language1)
        output_language = Language(language2)
    
    return input_language, output_language, pairs