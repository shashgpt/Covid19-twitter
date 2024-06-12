import numpy as np
import tensorflow as tf
import pickle
import string
import re
import os

class Word_vectors(object):
    def __init__(self, config):
        self.config = config

    def load_pre_trained_w2v(self):
        """
        load pre-trained w2v word vectors by google
        """
        pre_trained_word_vectors = {}
        with open("datasets/pre_trained_word_vectors/GoogleNews-vectors-negative300.bin", "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split()) # 3000000, 300
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = ''.join([x.decode('latin-1') for x in word])
                        break
                    if ch != b'\n':
                        word.append(ch)
                pre_trained_word_vectors[word] = np.frombuffer(f.read(binary_len), dtype='float32')   
        
        return pre_trained_word_vectors
    
    def create_vocabulary(self, dataset):
        """
        tokenize each sentence in dataset by sentence.split()
        assign each token in every sentence a unique int value (unique in the entire dataset)
        return a dictionary word_index[word] = unique int value
        """
        vectorize_layer = tf.keras.layers.TextVectorization(standardize=None, split='whitespace')
        # vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace')
        vectorize_layer.adapt(np.array(dataset["sentence"]))
        vocab = vectorize_layer.get_vocabulary()
        word_index = dict(zip(vocab, range(len(vocab))))
        return word_index

    def create_word_vectors(self, dataset=None):
        
        pre_trained_word_vectors = self.load_pre_trained_w2v()
        word_index = self.create_vocabulary(dataset)

        num_tokens = len(word_index) + 2
        embedding_dim = 300 # for w2v
        word_vectors = np.zeros((num_tokens, embedding_dim))
        hits = 0
        misses = 0  
        for word, i in word_index.items():
            embedding_vector = pre_trained_word_vectors.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                word_vectors[i] = embedding_vector
                hits += 1
            else:
                word_vectors[i] = np.random.uniform(-0.25, 0.25, embedding_dim)
                misses += 1
        print("\nWord vectors created")
        print("\nConverted %d words (%d misses)" % (hits, misses))
        
        return word_vectors, word_index