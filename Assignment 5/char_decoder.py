#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.

        char_embed = self.decoderCharEmb(input)
        dec_output, dec_hidden = self.charDecoder(char_embed, dec_hidden)
        scores = self.char_output_projection(dec_output)

        return scores, dec_hidden

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # input sequence -> x_1 ... x_n
        input = char_sequence[:-1]
        scores, dec_hidden = self.forward(input, dec_hidden)

        # target -> x_2 ... x_{n+1}
        target = char_sequence[1:]

        # reshaping for CE_Loss dimensions
        # scores: (length - 1, batch_size, self.vocab_size) -> (length - 1 * batch_size, self.vocab_size)
        # target: (length - 1, batch_size) -> (length - 1 * batch_size)
        scores = scores.view(-1, scores.shape[2])
        target = target.view(-1)

        # ignore pad_index = 0, sum of loss instead of average
        loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char2id['<pad>'])

        return loss(scores, target)

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates

        start_index = self.target_vocab.start_of_word
        end_index = self.target_vocab.end_of_word

        # input shape: (batch_size) -> (1, batch_size)
        input = [start_index for _ in range(batch_size)]
        input = torch.tensor(input, device=device).unsqueeze(dim=0)

        # output = [[''] for _ in range(batch_size)]
        decodedWords = [''] * batch_size

        for t in range(max_length):
            # scores shape: (1, batch_size, vocab_size)
            # input shape: (1, batch_size) using argmax and (batch_size) using squeeze
            scores, dec_hidden = self.forward(input, dec_hidden)
            input = torch.argmax(scores, dim=2).squeeze(dim=0)

            # appending char for each word to respective word
            decodedWords = [word + self.target_vocab.id2char[char_index] for word, char_index in zip(decodedWords, input.tolist())]
            input = input.unsqueeze(dim=0)


        # if word has '}' => truncate
        for i, word in enumerate(decodedWords):
            end_index = word.find('}')
            if end_index != -1:
                decodedWords[i] = word[:end_index]

        return decodedWords

        ### END YOUR CODE
