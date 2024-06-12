import os
import pickle
import pandas as pd
import tensorflow as tf
import string
import re
import numpy as np
import timeit
from tqdm import tqdm
import preprocessor as p


class Preprocess_dataset(object):
    def __init__(self, config):
        self.config = config
    
    def preprocess_text(self, text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
        """
        Preprocess text as per Keras Tokenizer preprocess code. 
        Tokenize by just sentence.split()
        Whole process is similar to Keras Tokenizer
        """
        p.set_options(p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
        cleaned_tweet = p.clean(text)
        text = cleaned_tweet.lower() # lower case
        maketrans = str.maketrans
        translate_dict = {c: split for c in filters}
        translate_map = maketrans(translate_dict) 
        text = text.translate(translate_map) # remove all punctuations and replace them with whitespace (because puntuations mark as a whitespace between words)
        return text
    
    def create_rule_masks(self, dataset):
        """
        create rule masks for each sentence in the dataset
        """
        rule_label_masks = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            try:
                if rule_label == 1 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("but")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["but"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 2 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("yet")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    rule_label_mask = [0]*len(a_part_tokenized_sentence) + [0]*len(["yet"]) + [1]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 3 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("though")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["though"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)

                elif rule_label == 4 and contrast == 1:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("while")+1:]
                    rule_label_mask = [1]*len(a_part_tokenized_sentence) + [0]*len(["while"]) + [0]*len(b_part_tokenized_sentence)
                    rule_label_masks.append(rule_label_mask)
                
                else:
                    mask_length = len(tokenized_sentence)
                    rule_label_mask = [1]*mask_length
                    rule_label_masks.append(rule_label_mask)
            except:
                mask_length = len(tokenized_sentence)
                rule_label_mask = [1]*mask_length
                rule_label_masks.append(rule_label_mask)
        
        dataset["rule_mask"] = rule_label_masks
        return dataset

    def preprocess_covid_tweets(self, dataset):
            
        # Select columns
        dataset = dataset[['tweet','sentiment_label', 'rule_label', 'contrast']]

        # Select no rule, but, yet, though and while rule sentences
        dataset = dataset.loc[(dataset["rule_label"]=="not_applicable")|(dataset["rule_label"]=="A-but-B")|(dataset["rule_label"]=="A-yet-B")|(dataset["rule_label"]=="A-though-B")|(dataset["rule_label"]=="A-while-B")]
        
        # Converting str values to int values for rule label and contrast columns
        dataset['sentiment_label'] = dataset['sentiment_label'].map({'negative': 0, 'positive': 1})
        dataset['rule_label'] = dataset['rule_label'].map({'not_applicable': 0, 'A-but-B': 1, 'A-yet-B': 2, 'A-though-B': 3, 'A-while-B': 4})
        dataset['contrast'] = dataset['contrast'].map({'not_applicable': 0, 'no_contrast': 0, 'contrast': 1})

        # Renaming the preprocessed_tweet column to sentence
        dataset = dataset.rename(columns={'tweet':'sentence'})

        # Preprocess sentences (VECTORIZE)
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence in list(dataset['sentence'])]
        dataset["sentence"] = preprocessed_sentences
        
        # Create rule masks (VECTORIZE)
        dataset = self.create_rule_masks(dataset)

        return dataset