from functools import update_wrapper
from scipy.stats import ttest_ind
import math
import numpy as np
import pandas as pd
import random

def zero_scores_for_punctuations(tokens, scores):
    for index, token in enumerate(tokens):
        if token == "," or token == '`' or token == "'":
            try:
                scores[index] = 0
            except:
                continue
    return scores

def calculate_percy(results_one_rule, results_explanations, type_of_percy=None, K=5):
    EA_values = {
                "one_rule":[],
                "one_rule_contrast":[],
                "one_rule_no_contrast":[]
                }
    counter = 0
    results_one_rule = pd.DataFrame(results_one_rule)
    results_explanations = pd.DataFrame(results_explanations)
    
    sentences = list(results_explanations['sentence'])
    sent_predictions = list(results_one_rule['sentiment_prediction_output'])
    sent_labels = list(results_one_rule['sentiment_label'])
    rule_labels = list(results_one_rule['rule_label'])
    contrasts = list(results_one_rule['contrast'])
    
    try:
        features = list(results_explanations["features"])
    except:
        features = list(results_explanations["sentence"])
    explanations = list(results_explanations["LIME_explanation_normalised"])

    # Anecdotal examples
    corr_sent_pred_wrong_percy_score = {"sentence":[],
                                        "sentiment":[],
                                        "a_conjunct":[],
                                        "b_conjunct":[],
                                        "a_conjunct_explanations":[],
                                        "b_conjunct_explanations":[],
                                        "a_conjunct_score":[],
                                        "b_conjunct_score":[]}

    for index, sentence in enumerate(sentences):
        
        # Select LIME explanations corresponding to those tokens
        exp = explanations[index]
        
        # Check 1: Drop the sentences for which LIME explanation couldn't be calculated
        if explanations[index] == "couldn't process":
            continue

        # Check 2: If A&B conjuncts contains 1 token atleast
        tokenized_sentence = sentence
        if rule_labels[index] == 1:
            rule_word = "but"
        elif rule_labels[index] == 2:
            rule_word = "yet"
        elif rule_labels[index] == 3:
            rule_word = "though"
        elif rule_labels[index] == 4:
            rule_word = "while"
        rule_word_index = tokenized_sentence.index(rule_word)
        A_conjunct = tokenized_sentence[:rule_word_index]
        B_conjunct = tokenized_sentence[rule_word_index+1:len(tokenized_sentence)]
        A_conjunct_exp = exp[0:rule_word_index]
        B_conjunct_exp = exp[rule_word_index+1:len(tokenized_sentence)]
        if len(A_conjunct) == 0 or len(B_conjunct) == 0 or len(A_conjunct_exp) == 0 or len(B_conjunct_exp)==0:
            continue
        
        # Select the tokens in Conjunct for P-value test (Fixing the number of tokens)
        A_conjunct_selected = []
        B_conjunct_selected = []
        A_conjunct_exp_sorted = sorted(A_conjunct_exp, reverse=True) # sorting tokens in descending order
        B_conjunct_exp_sorted = sorted(B_conjunct_exp, reverse=True)
        A_conjunct_exp_tokens = A_conjunct_exp_sorted[0:len(A_conjunct_exp_sorted)]
        B_conjunct_exp_tokens = B_conjunct_exp_sorted[0:len(B_conjunct_exp_sorted)]
        for value_index, value in enumerate(A_conjunct_exp_tokens):
            A_conjunct_selected.append(A_conjunct[A_conjunct_exp.index(value)])
        for value_index, value in enumerate(B_conjunct_exp_tokens):
            B_conjunct_selected.append(B_conjunct[B_conjunct_exp.index(value)])
        p_value = ttest_ind(A_conjunct_exp_tokens, B_conjunct_exp_tokens)[1] # Pvalue test to reject the null hypothesis (How does it apply in LIME-scores?)
        scores = exp

        # Calculating the PERCY score
        if rule_word == "but" or rule_word == "yet":
            if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) < np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                EA_value = 1
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
            else:
                EA_value = 0
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
        elif rule_word == "though" or rule_word == "while":
            if sent_predictions[index] == sent_labels[index] and np.mean(A_conjunct_exp_tokens) > np.mean(B_conjunct_exp_tokens) and p_value < 0.05:
                EA_value = 1
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
            else:
                EA_value = 0
                EA_values["one_rule"].append(EA_value)
                EA_values["one_rule_contrast"].append(EA_value)
                EA_values["one_rule_no_contrast"].append(EA_value)
    
    return EA_values