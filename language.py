"""
Language class for one-hot vector representations of words
"""

SOS_TOKEN = 0
EOS_TOKEN = 1

class Language:
    
    def __init__(self, name):
        """
        Attributes
            name        Name of language
            word2index  Mapping of word -> index
            index2word  Mapping of index -> word
            word2count  Counts (frequencies) of each word;
                          used for replacing rare words in language (i.e. lower frequencies)
        """
        self.name = name
        self.word2index = {}
        self.index2word = {
            0: "SOS", 
            1: "EOS"
        }
        self.word2count = {}
        self.num_words = 2
    
    def add_word(self, word):
        """
        Adds word to language
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        """
        Adds all words in a sentence to language
        """
        for word in sentence.split(' '):
            self.add_word(word)
