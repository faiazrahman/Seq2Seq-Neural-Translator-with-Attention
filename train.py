from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from data_loader import prepare_data
from language import Language
from network import EncoderRNN, DecoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = 0
EOS_TOKEN = 1

"""
Prepare training data
"""

input_language, output_language, pairs = prepare_data('fra', 'eng', reverse=False)

def tensors_from_pair(pair):
    """
    Generates an input tensor and target tensor for each pair of sentences (from input
    language to target language)
    """

    def indexes_from_sentence(language, sentences):
        return [language.word2index[word] for word in sentence.split(' ')]

    def tensor_from_sentence(language, sentence):
        indexes = indexes_from_sentence(language, sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, 
                            dtype=torch.long, 
                            device=device).view(-1, 1)

    input_tensor = tensor_from_sentence(input_language, pair[0])
    target_tensor = tensor_from_sentence(output_language, pair[1])
    return (input_tensor, target_tensor)
