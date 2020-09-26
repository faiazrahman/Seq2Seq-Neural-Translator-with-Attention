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

"""
The teacher forcing algorithm trains recurrent networks by supplying observed
sequence values as inputs during training and using the network’s own one-step-ahead 
predictions to do multi-step sampling [Ref: https://arxiv.org/abs/1610.09038]. 
Thus, it is essentially a method in which a neural model makes a decision conditioned 
on the gold history of the target sequence. This can lead to quicker convergence.

Additional Refs: https://arxiv.org/abs/1906.07651, https://arxiv.org/abs/1907.08506
"""

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        # If not using teacher forcing, use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

