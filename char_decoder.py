#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
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
        # YOUR CODE HERE for part 2a
        # TODO - Initialize as an nn.Module.
        # - Initialize the following variables:
        # self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        # self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        # self.decoderCharEmb: Embedding matrix of character embeddings
        # self.target_vocab: vocabulary for the target language
        ###
        # Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        # - Set the padding_idx argument of the embedding matrix.
        # - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()

        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(
            hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(
            len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.celoss = nn.CrossEntropyLoss(
            ignore_index=target_vocab.char2id['<pad>'], reduction='sum')
        # END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        # YOUR CODE HERE for part 2b
        # TODO - Implement the forward pass of the character decoder.

        emb = self.decoderCharEmb(input)
        # print('emb size', emb.size())
        output, dec_hidden_out = self.charDecoder(emb, dec_hidden)
        # print('output size', output.size(), dec_hidden_out[0].size())

        scores = self.char_output_projection(output)
        # print('scores size', scores.size())
        return scores, dec_hidden_out
        # END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        # YOUR CODE HERE for part 2c
        # TODO - Implement training forward pass.
        ###
        # Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        # - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, dec_hidden_out = self.forward(
            char_sequence, dec_hidden=dec_hidden)
        # print(p_t.size(), char_sequence.size())
        length, batch, vocab_size = list(scores.size())
        new_char_sequence = char_sequence[1:].contiguous().view(-1)
        # print('new_char_sequence', new_char_sequence.size())
        p_t = scores[:-1].view((length-1)*batch, vocab_size)
        # print('p_t', p_t.size())

        loss = self.celoss(p_t, new_char_sequence)
        return loss
        # END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        # YOUR CODE HERE for part 2d
        # TODO - Implement greedy decoding.
        # Hints:
        # - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        # - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        # - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        # Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        output_word = []
        _, batch, hidden_size = list(initialStates[0].size())
        current_chars = torch.tensor(
            [self.target_vocab.char2id['{']] * batch, dtype=torch.long, device=device).unsqueeze(dim=0)

        current_state = initialStates
        for _ in range(max_length):
            scores, dec_hidden_out = self.forward(
                current_chars, dec_hidden=current_state)
            p_t = nn.functional.softmax(scores, dim=2)
            characters = p_t.argmax(dim=2)
            char_characters = [self.target_vocab.id2char[k]
                               for k in characters.squeeze(0).tolist()]
            output_word.append(char_characters)
            current_state = dec_hidden_out
            current_chars = characters

        output_word = list(map(list, zip(*output_word)))
        output_word = [''.join(w) for w in output_word]
        output_word = [s.split('}', 1)[0] for s in output_word]
        return output_word

        # END YOUR CODE
