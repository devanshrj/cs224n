#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, dropout_rate=0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.char_embed_size = 50               # hard coded, since given
        self.word_embed_size = word_embed_size
        self.vocab = vocab

        self.dropout = nn.Dropout(dropout_rate)
        self.embeddings = nn.Embedding(len(self.vocab.char2id), self.char_embed_size, padding_idx=vocab.char2id['<pad>'])

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        sentence_length, batch_size, max_word_length = input.shape

        # step 2 -> embedding lookup
        # x_char_embed shape -> (sentence_length, batch_size, max_word_length, char_embed_size)
        # x_reshaped -> (sentence_length * batch_size, char_embed_size, max_word_length)
        x_char_embed = self.embeddings(input)
        x_reshaped = x_char_embed.reshape(-1, self.char_embed_size, max_word_length)

        # step 3 -> convolutional layer
        self.CNN = CNN(self.char_embed_size, self.word_embed_size, max_word_length)
        x_conv_out = self.CNN(x_reshaped)

        # step 4 -> highway network
        self.Highway = Highway(self.word_embed_size)
        x_highway = self.Highway(x_conv_out)

        x_word_embed = self.dropout(x_highway)
        x_word_embed = x_word_embed.reshape(sentence_length, -1, self.word_embed_size)

        return x_word_embed

        ### END YOUR CODE

