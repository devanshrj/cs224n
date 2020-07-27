#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""
import sys

import torch
import torch.nn as nn

class CNN(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size, word_embed_size, max_word_length, kernel_size=5):
    	"""Init CNN layer.

    	@params char_embed_size (int): Size of char embeddings, for input channels
    	@params word_embed_size (int): Size of word embeddings, for output channels
        @params max_word_length (int): Max length of word in a batch 

    	input: batch_size, char_embed_size, max_word_length
    	output: batch_size, word_embed_size, max_word_length - kernel_size + 1

    	Conv1D layer
    	Max pool layer
    	"""
    	super(CNN, self).__init__()				# calling Module's init
    	self.input_channels  = char_embed_size	# input channels
    	self.num_filters 	 = word_embed_size	# output channels
    	self.kernel_size	 = kernel_size
    	self.m_word          = max_word_length

    	self.conv1D = nn.Conv1d(in_channels=self.input_channels, out_channels=self.num_filters, kernel_size=5, padding=1, bias=True)
    	# kernel size for max pool = m_word - kernel_size + 1 + 2 * padding
    	self.max_pool = nn.MaxPool1d(kernel_size=(self.m_word - kernel_size + 3))


    def forward(self, x_reshaped):
    	"""Forward propagation.

    	@param x_reshaped (Tensor): Tensor of character embeddings with shape (batch_size, char_embed_size, max_word_length)

		@returns x_conv_out (Tensor): Output from conv layer with shape (batch_size, word_embed_size)
    	"""

    	x_conv = self.conv1D(x_reshaped)
    	x_conv_out = self.max_pool(torch.relu(x_conv)).squeeze()

    	return x_conv_out

    ### End of CNN Class


def main():
    """ Main func."""

	# Check Python & PyTorch Versions
    assert(sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    print('-' * 80)
    print("Running Sanity Check for Question 1g: CNN")
    print('-' * 80)

    print("Running test on a list of words")
    test_values = [[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]], 
    			   [[18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35]], 
    			   [[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53]], 
    			   [[54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65], [66, 67, 68, 69, 70, 71]]]

    test_reshape = torch.tensor(test_values).float()

    BATCH_SIZE = 4
    MAX_WORD_LENGTH = 6
    char_size = 3
    word_size = 5

    model = CNN(char_size, word_size, MAX_WORD_LENGTH)
    output = model(test_reshape)
    output_expected_size = [BATCH_SIZE, word_size]
    assert list(output.size()) == output_expected_size, "output shape is incorrect"

    print("Sanity Check Passed for Question 1g: CNN")
    print('-' * 80)


if __name__ == '__main__':
    main()