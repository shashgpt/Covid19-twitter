from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf

class Corrects_distribution(object):

    def __init__(self, len_test_data_size):
        self.len_test_data_size = len_test_data_size

    def model_sentiment_correct_distributions(self, results):

        model_sentiment_corrects = {'overall':None, 
                                    'no_rule':None,
                                    'one_rule':None,
                                    'one_rule_contrast':None,
                                    'one_rule_no_contrast':None,
                                    'a_but_b':None, 
                                    'a_but_b_contrast':None, 
                                    'a_but_b_no_contrast':None,
                                    'a_yet_b':None, 
                                    'a_yet_b_contrast':None, 
                                    'a_yet_b_no_contrast':None,
                                    'a_though_b':None, 
                                    'a_though_b_contrast':None, 
                                    'a_though_b_no_contrast':None,
                                    'a_while_b':None, 
                                    'a_while_b_contrast':None, 
                                    'a_while_b_no_contrast':None
                                    }
        
        results = pd.DataFrame(results)

        # Overall
        n1 = np.array(results['sentiment_label'])
        n2 = np.array(results['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["overall"] = n

        # No rule
        n1 = np.array(results.loc[results["rule_label"]==0]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]==0]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["no_rule"] = n

        # One rule
        n1 = np.array(results.loc[results["rule_label"]!=0]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]!=0]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["one_rule"] = n

        # One rule, Contrast
        n1 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["one_rule_contrast"] = n

        # One rule, No Contrast
        n1 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["one_rule_no_contrast"] = n

        # A-but-B rule
        n1 = np.array(results.loc[results["rule_label"]==1]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]==1]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_but_b"] = n

        # A-but-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==1)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==1)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_but_b_contrast"] = n

        # A-but-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==0)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==0)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_but_b_no_contrast"] = n

        # A-yet-B rule
        n1 = np.array(results.loc[results["rule_label"]==2]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]==2]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_yet_b"] = n

        # A-yet-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==1)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==1)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_yet_b_contrast"] = n

        # A-yet-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==0)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==0)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_yet_b_no_contrast"] = n

        # A-though-B rule
        n1 = np.array(results.loc[results["rule_label"]==3]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]==3]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_though_b"] = n

        # A-though-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==1)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==1)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_though_b_contrast"] = n

        # A-though-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==0)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==0)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_though_b_no_contrast"] = n

        # A-while-B rule
        n1 = np.array(results.loc[results["rule_label"]==4]['sentiment_label'])
        n2 = np.array(results.loc[results["rule_label"]==4]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_while_b"] = n

        # A-while-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==1)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==1)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_while_b_contrast"] = n

        # A-while-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==0)]['sentiment_label'])
        n2 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==0)]['sentiment_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_sentiment_corrects["a_while_b_no_contrast"] = n

        return model_sentiment_corrects

    def mask_model_rule_mask_correct_distributions(self, results_mask):

        rule_mask_corrects = {'overall':None, 
                            'no_rule':None,
                            'one_rule':None,
                            'one_rule_contrast':None,
                            'one_rule_no_contrast':None,
                            'a_but_b':None, 
                            'a_but_b_contrast':None, 
                            'a_but_b_no_contrast':None,
                            'a_yet_b':None, 
                            'a_yet_b_contrast':None, 
                            'a_yet_b_no_contrast':None,
                            'a_though_b':None, 
                            'a_though_b_contrast':None, 
                            'a_though_b_no_contrast':None,
                            'a_while_b':None, 
                            'a_while_b_contrast':None, 
                            'a_while_b_no_contrast':None
                            }
        
        results_mask = pd.DataFrame(results_mask)

        # Overall
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['overall'] = n

        # No rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==0]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==0]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['no_rule'] = n

        # One rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]!=0]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]!=0]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['one_rule'] = n

        # One rule, contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==1)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==1)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['one_rule_contrast'] = n

        # One rule, no contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==0)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]!=0)&(results_mask["contrast"]==0)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['one_rule_no_contrast'] = n

        # A-but-B rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==1]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==1]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_but_b'] = n

        # A-but-B rule, contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==1) & (results_mask["contrast"]==1)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==1) & (results_mask["contrast"]==1)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_but_b_contrast'] = n

        # A-but-B rule, no contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==1) & (results_mask["contrast"]==0)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==1) & (results_mask["contrast"]==0)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_but_b_no_contrast'] = n

        # A-yet-B rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==2]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==2]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_yet_b'] = n

        # A-yet-B rule, contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==2) & (results_mask["contrast"]==1)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==2) & (results_mask["contrast"]==1)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_yet_b_contrast'] = n

        # A-yet-B rule, no contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==2) & (results_mask["contrast"]==0)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==2) & (results_mask["contrast"]==0)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_yet_b_no_contrast'] = n

        # A-though-B rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==3]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==3]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_though_b'] = n

        # A-though-B rule, contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==3) & (results_mask["contrast"]==1)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==3) & (results_mask["contrast"]==1)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_though_b_contrast'] = n

        # A-though-B rule, no contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==3) & (results_mask["contrast"]==0)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==3) & (results_mask["contrast"]==0)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_though_b_no_contrast'] = n

        # A-while-B rule
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==4]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[results_mask["rule_label"]==4]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_while_b'] = n

        # A-while-B rule, contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==4) & (results_mask["contrast"]==1)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==4) & (results_mask["contrast"]==1)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_while_b_contrast'] = n

        # A-while-B rule, no contrast
        n1 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==4) & (results_mask["contrast"]==0)]["rule_label_mask"], value=5, padding='post')
        n2 = tf.keras.preprocessing.sequence.pad_sequences(results_mask.loc[(results_mask["rule_label"]==4) & (results_mask["contrast"]==0)]["rule_label_mask_prediction_output"], value=5, padding='post')
        l = (n1==n2)
        n = np.array([l[i].all() for i in range(l.shape[0])])
        n = np.where(n == True, 1, 0)
        n = n.tolist()
        rule_mask_corrects['a_while_b_no_contrast'] = n
        
        return rule_mask_corrects

    def mask_model_contrast_correct_distributions(self, results):

        model_contrast_corrects = {'overall':None, 
                                    'no_rule':None,
                                    'one_rule':None,
                                    'one_rule_contrast':None,
                                    'one_rule_no_contrast':None,
                                    'a_but_b':None, 
                                    'a_but_b_contrast':None, 
                                    'a_but_b_no_contrast':None,
                                    'a_yet_b':None, 
                                    'a_yet_b_contrast':None, 
                                    'a_yet_b_no_contrast':None,
                                    'a_though_b':None, 
                                    'a_though_b_contrast':None, 
                                    'a_though_b_no_contrast':None,
                                    'a_while_b':None, 
                                    'a_while_b_contrast':None, 
                                    'a_while_b_no_contrast':None
                                    }
        
        results = pd.DataFrame(results)

        # Overall
        n1 = np.array(results['contrast'])
        n2 = np.array(results['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["overall"] = n

        # No rule
        n1 = np.array(results.loc[results["rule_label"]==0]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]==0]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["no_rule"] = n

        # One rule
        n1 = np.array(results.loc[results["rule_label"]!=0]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]!=0]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["one_rule"] = n

        # One rule, Contrast
        n1 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["one_rule_contrast"] = n

        # One rule, No Contrast
        n1 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["one_rule_no_contrast"] = n

        # A-but-B rule
        n1 = np.array(results.loc[results["rule_label"]==1]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]==1]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_but_b"] = n

        # A-but-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==1)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==1)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_but_b_contrast"] = n

        # A-but-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==0)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==1) & (results["contrast"]==0)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_but_b_no_contrast"] = n

        # A-yet-B rule
        n1 = np.array(results.loc[results["rule_label"]==2]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]==2]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_yet_b"] = n

        # A-yet-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==1)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==1)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_yet_b_contrast"] = n

        # A-yet-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==0)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==2) & (results["contrast"]==0)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_yet_b_no_contrast"] = n

        # A-though-B rule
        n1 = np.array(results.loc[results["rule_label"]==3]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]==3]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_though_b"] = n

        # A-though-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==1)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==1)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_though_b_contrast"] = n

        # A-though-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==0)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==3) & (results["contrast"]==0)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_though_b_no_contrast"] = n

        # A-while-B rule
        n1 = np.array(results.loc[results["rule_label"]==4]['contrast'])
        n2 = np.array(results.loc[results["rule_label"]==4]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_while_b"] = n

        # A-while-B rule, contrast
        n1 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==1)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==1)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_while_b_contrast"] = n

        # A-while-B rule, no contrast
        n1 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==0)]['contrast'])
        n2 = np.array(results.loc[(results["rule_label"]==4) & (results["contrast"]==0)]['contrast_prediction_output'])
        l = (n1==n2)
        n = np.where(l == True, 1, 0)
        n = n.tolist()
        model_contrast_corrects["a_while_b_no_contrast"] = n

        return model_contrast_corrects

    def model_lime_correct_distributions(self, results_one_rule, lime_explanations):

        model_lime_corrects = {'one_rule':[],
                                'one_rule_contrast':[],
                                'one_rule_no_contrast':[],
                                'a_but_b':[],
                                'a_but_b_contrast':[], 
                                'a_but_b_no_contrast':[],
                                'a_yet_b':[],
                                'a_yet_b_contrast':[], 
                                'a_yet_b_no_contrast':[],
                                'a_though_b':[],
                                'a_though_b_contrast':[], 
                                'a_though_b_no_contrast':[],
                                'a_while_b':[],
                                'a_while_b_contrast':[], 
                                'a_while_b_no_contrast':[]
                                }
        
        # results = pd.DataFrame(results)
        results_one_rule = pd.DataFrame(results_one_rule)
        lime_explanations = pd.DataFrame(lime_explanations)

        # # One rule data
        # sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
        # rule_labels = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['rule_label']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['rule_label'])
        # contrasts = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['contrast']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['contrast'])
        # predictions = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_prediction_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_prediction_output'])
        # sentiments = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_label']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_label'])
        sentences = list(results_one_rule['sentence'])
        predictions = list(results_one_rule['sentiment_prediction_output'])
        sentiments = list(results_one_rule['sentiment_label'])
        rule_labels = list(results_one_rule['rule_label'])
        contrasts = list(results_one_rule['contrast'])

        for index, explanation in enumerate(lime_explanations["LIME_explanation_normalised"]):
            tokenized_sentence = lime_explanations['sentence'][index]
            sentiment = sentiments[index]
            prediction = predictions[index]
            rule_label = rule_labels[index]
            contrast = contrasts[index]

            if explanation == "couldn't process":
                continue

            if rule_label == 1 and contrast == 1 and sentiment == prediction:
                if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1):
                    word_index_value = tokenized_sentence.index('but')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B > sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_contrast"].append(1)
                        model_lime_corrects["a_but_b"].append(1)
                        model_lime_corrects["a_but_b_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_contrast"].append(0)
                        model_lime_corrects["a_but_b"].append(0)
                        model_lime_corrects["a_but_b_contrast"].append(0)
                else:
                    print(tokenized_sentence)
                    
            elif rule_label == 1 and contrast == 0 and sentiment == prediction:
                if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1):
                    word_index_value = tokenized_sentence.index('but')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B > sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_no_contrast"].append(1)
                        model_lime_corrects["a_but_b"].append(1)
                        model_lime_corrects["a_but_b_no_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_no_contrast"].append(0)
                        model_lime_corrects["a_but_b"].append(0)
                        model_lime_corrects["a_but_b_no_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 2 and contrast == 1 and sentiment == prediction:
                if ('yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1):
                    word_index_value = tokenized_sentence.index('yet')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B > sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_contrast"].append(1)
                        model_lime_corrects["a_yet_b"].append(1)
                        model_lime_corrects["a_yet_b_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_contrast"].append(0)
                        model_lime_corrects["a_yet_b"].append(0)
                        model_lime_corrects["a_yet_b_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 2 and contrast == 0 and sentiment==prediction:
                if ('yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1):
                    word_index_value = tokenized_sentence.index('yet')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B > sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_no_contrast"].append(1)
                        model_lime_corrects["a_yet_b"].append(1)
                        model_lime_corrects["a_yet_b_no_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_no_contrast"].append(0)
                        model_lime_corrects["a_yet_b"].append(0)
                        model_lime_corrects["a_yet_b_no_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 3 and contrast == 1 and sentiment==prediction:
                if ('though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
                    word_index_value = tokenized_sentence.index('though')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B < sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_contrast"].append(1)
                        model_lime_corrects["a_though_b"].append(1)
                        model_lime_corrects["a_though_b_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_contrast"].append(0)
                        model_lime_corrects["a_though_b"].append(0)
                        model_lime_corrects["a_though_b_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 3 and contrast == 0 and sentiment==prediction:
                if ('though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
                    word_index_value = tokenized_sentence.index('though')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B < sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_no_contrast"].append(1)
                        model_lime_corrects["a_though_b"].append(1)
                        model_lime_corrects["a_though_b_no_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_no_contrast"].append(0)
                        model_lime_corrects["a_though_b"].append(0)
                        model_lime_corrects["a_though_b_no_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 4 and contrast == 1 and sentiment==prediction:
                if ('while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
                    word_index_value = tokenized_sentence.index('while')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B < sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_contrast"].append(1)
                        model_lime_corrects["a_while_b"].append(1)
                        model_lime_corrects["a_while_b_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_contrast"].append(0)
                        model_lime_corrects["a_while_b"].append(0)
                        model_lime_corrects["a_while_b_contrast"].append(0)
                else:
                    print(tokenized_sentence)
            
            elif rule_label == 4 and contrast == 0 and sentiment==prediction:
                if ('while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
                    word_index_value = tokenized_sentence.index('while')
                    sum_A = sum(explanation[:word_index_value])
                    sum_B = sum(explanation[word_index_value+1:])
                    if sum_B < sum_A:
                        model_lime_corrects["one_rule"].append(1)
                        model_lime_corrects["one_rule_no_contrast"].append(1)
                        model_lime_corrects["a_while_b"].append(1)
                        model_lime_corrects["a_while_b_no_contrast"].append(1)
                    else:
                        model_lime_corrects["one_rule"].append(0)
                        model_lime_corrects["one_rule_no_contrast"].append(0)
                        model_lime_corrects["a_while_b"].append(0)
                        model_lime_corrects["a_while_b_no_contrast"].append(0)
                else:
                    print(tokenized_sentence)

        return model_lime_corrects