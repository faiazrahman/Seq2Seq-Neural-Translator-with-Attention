"""
Data loader for sentence pairs between two languages, with preprocessing
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

from language import Language

PATH_TO_DATA = '../data/'
MAX_LENGTH = 10

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

    print("Reading lines...")
    
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

def filter_pairs(pairs):
    """
    Filters sentence pairs to only include those under MAX_LENGTH limit,
    to reduce training time
    """

    def filter_pair(p):
        """
        Filters to include sentences that translate to the following forms
        """

        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )

        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)
    
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(language1, language2, reverse=False):
    """
    Prepares the data by splitting each line into sentence pairs 
    from language1 to language2, normalizing text and filtering by
    length and content, and then constructing Language classes' word2index
    mappings
    """
    
    input_language, output_language, pairs = read_languages(language1, language2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_language.add_sentence(pair[0])
        output_language.add_sentence(pair[1])
    print("Counted words:")
    print(input_language.name, input_language.num_words)
    print(output_language.name, output_language.num_words)
    return input_language, output_language, pairs