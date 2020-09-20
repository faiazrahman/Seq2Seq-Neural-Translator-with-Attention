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

