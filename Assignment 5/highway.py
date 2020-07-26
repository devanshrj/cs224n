#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""
import sys

import torch
import torch.nn as nn

class Highway(nn.Module):
	# pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embed_size):
    	""" Init Highway Network.

    	@param word_embed_size (int): Size of word embeddings, output of conv layer and highway network

    	self.projection (Linear layer) -> ReLU
    	self.gate (Linear layer) -> sigmoid
    	self.dropout(Dropout layer)
    	"""
    	super(Highway, self).__init__()		# calling Module's init
    	self.word_embed_size = word_embed_size

    	self.projection = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
    	self.gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)


    def forward(self, x_conv_out):
    	""" Forward propagation.

    	@param x_conv (Tensor): Output from convolutional layer with shape (batch_size, word_embed_size)

		@returns x_word_embed (Tensor): Output from highway network with shape (batch_size, word_embed_size)
    	"""
    	
    	x_projection = torch.relu(self.projection(x_conv_out))
    	x_gate = torch.sigmoid(self.gate(x_conv_out))

    	x_highway = x_gate * x_projection + (1 - x_gate) * x_conv_out

    	return x_highway

    ### End of Highway Class

def main():
    """ Main func."""

	# Check Python & PyTorch Versions
    assert(sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    print('-' * 80)
    print("Running Sanity Check for Question 1f: Highway Network")
    print('-' * 80)

    print("Running test on a list of word vectors")
    test_values = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
    test_conv = torch.tensor(test_values).float()
    BATCH_SIZE = 4
    word_size = 5

    model = Highway(word_embed_size=word_size)
    output = model(test_conv)
    output_expected_size = [BATCH_SIZE, word_size]
    assert list(output.size()) == output_expected_size, "output shape is incorrect"

    print("Sanity Check Passed for Question 1f: Highway Network")
    print('-' * 80)


if __name__ == '__main__':
    main()