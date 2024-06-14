import os
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from lime import lime_text
import traceback
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2Config

from scripts.additional_validation_sets import AdditionalValidationSets
from scripts.corrects_distribution import Corrects_distribution
from scripts.percy_score import calculate_percy

class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def compute_mask(self, *args, **kwargs):
        return self.token_emb.compute_mask(*args, **kwargs)

    def call(self, input_ids, word_embeddings):
        maxlen = tf.shape(input_ids)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return word_embeddings + positions

class gpt2_transformer(Model):
    def __init__(self, config, num_heads, maxlen, epsilon, **kwargs):
        super().__init__()
        self.config = config
        self.configuration = GPT2Config().to_dict()
        self.encoder = TFGPT2Model.from_pretrained('gpt2')
        for layer in self.encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        self.positional_embedding_layer = PositionalEmbedding(maxlen=maxlen, 
                                                              vocab_size=self.configuration["vocab_size"], 
                                                              embed_dim=self.configuration["n_embd"])
        self.att = layers.MultiHeadAttention(num_heads=num_heads, 
                                             key_dim=self.configuration["n_embd"])
        self.ffn = tf.keras.Sequential([layers.Dense(num_heads*4, activation="relu"), 
                                        layers.Dense(self.configuration["n_embd"])])
        self.layernorm = layers.LayerNormalization(epsilon=epsilon)
        self.dropout = layers.Dropout(self.config["dropout"])
        self.global_average_pooling_1d = layers.GlobalAveragePooling1D()
        self.add = tf.keras.layers.Add()
        self.dense = tf.keras.layers.Dense(20, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    
    def call(self, input_ids, training, **kwargs):

        #bert_tweet output
        word_embeddings = self.encoder(input_ids).last_hidden_state
       
        #positional embeddings
        positional_embeddings = self.positional_embedding_layer(input_ids, word_embeddings)
        positional_embeddings = self.dropout(positional_embeddings, training=training)
        positional_embeddings = self.layernorm(positional_embeddings)

        #encoder
        encoder_output = self.att(positional_embeddings, positional_embeddings)
        encoder_output = self.dropout(encoder_output, training=training)
        encoder_output = self.layernorm(positional_embeddings + encoder_output)
        encoder_output_ffn = self.ffn(encoder_output)
        encoder_output_ffn = self.dropout(encoder_output_ffn, training=training)
        encoder_output = self.layernorm(encoder_output + encoder_output_ffn)

        #output (taking average of all hidden states)
        output = self.global_average_pooling_1d(encoder_output)
        output = self.dropout(output)
        # output = self.dense(output)
        # output = self.dropout(output)
        output = self.out(output)
        return output
    
    def build_model(self, input_shape):
        input_data = layers.Input(shape=(input_shape,), dtype="float32")
        model = tf.keras.Model(inputs=input_data, outputs=self.call(input_data, training=False))
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy'])
        model.summary()
        return model

class train_gpt2_transformer(object):
    def __init__(self, config):
        self.config = config

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        max_len = 0
        input_ids = []
        for sentence in sentences:
            tokenized_context = tokenizer.encode(sentence)
            input_id = tokenized_context
            input_ids.append(input_id)
            if len(input_id) > max_len:
                max_len = len(input_id)
        for index, input_id in enumerate(input_ids):
            padding_length = max_len - len(input_ids[index])
            input_ids[index] = input_ids[index] + ([0] * padding_length)
        return np.array(input_ids)

    def pad(self, sentences, maxlen):
        """
        right pad sequence with 0 till max token length sentence
        """
        return tf.keras.utils.pad_sequences(sentences, value=0, padding='post', maxlen=maxlen)

    def prediction(self, text):
        x = self.vectorize(text)
        x = self.pad(x, self.maxlen)
        pred_prob_1 = self.model.predict(x, batch_size=1000)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def train_model(self, train_dataset, val_datasets, test_datasets):

        #make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")
        
        #create train, val, and test datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])
        test_sentences = self.vectorize(test_datasets["test_dataset"]["sentence"])
        test_sentiment_labels = np.array(test_datasets["test_dataset"]["sentiment_label"])
        maxlen = max([train_sentences.shape[1], val_sentences.shape[1], test_sentences.shape[1]])
        self.maxlen = maxlen
        train_sentences = self.pad(train_sentences, maxlen)
        val_sentences = self.pad(val_sentences, maxlen)
        test_sentences = self.pad(test_sentences, maxlen)
        train_dataset = (train_sentences, train_sentiment_labels)
        val_dataset = (val_sentences, val_sentiment_labels)
        test_dataset = (test_sentences, test_sentiment_labels)

        #create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        #define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',              # 1. Calculate val_loss_1 
                                                                    min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                    patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                    verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                    mode="min",
                                                                    baseline=None, 
                                                                    restore_best_weights=True)
        my_callbacks = [
                        early_stopping_callback, 
                        AdditionalValidationSets(additional_validation_datasets, self.config)
                       ]
        
        #model compilation and summarization
        model = gpt2_transformer(self.config,
                                     maxlen=self.maxlen,
                                     num_heads=self.config["hidden_units"],
                                     epsilon=1e-6)
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy'])
        # model.summary()
        self.model = model

        #train the model
        if self.config["train_model"] == True:
            self.model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    batch_size=self.config["mini_batch_size"], 
                    epochs=self.config["train_epochs"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks)

            #Save trained model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            self.model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        
        if self.config["evaluate_model"] == True:

            #load model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #Results to be created after evaluation
            results = {'sentence':[], 
                        'sentiment_label':[],  
                        'rule_label':[],
                        'contrast':[],
                        'sentiment_probability_output':[], 
                        'sentiment_prediction_output':[]}

            #Evaluation and predictions
            evaluations = self.model.evaluate(x=test_dataset[0], y=test_dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = self.model.predict(x=test_dataset[0])
            print(len(predictions))

            #Create results
            for index, sentence in enumerate(test_datasets["test_dataset"]["sentence"]):
                results['sentence'].append(test_datasets["test_dataset"]['sentence'][index])
                results['sentiment_label'].append(test_datasets["test_dataset"]['sentiment_label'][index])
                results['rule_label'].append(test_datasets["test_dataset"]['rule_label'][index])
                results['contrast'].append(test_datasets["test_dataset"]['contrast'][index])
            for prediction in predictions:
                results['sentiment_probability_output'].append(prediction)
                prediction = np.rint(prediction)
                results['sentiment_prediction_output'].append(prediction[0])

            #save the results
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)

            #Check the rule-subset of the results: balancing between contrast and no-contrast tweets of rule-subset
            results = pd.DataFrame(results)
            one_rule = pd.concat([results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)], results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]])
            one_rule = one_rule.reset_index(drop=True)
            one_rule_contrast_pos = one_rule.loc[(one_rule["contrast"]==1)&(one_rule["sentiment_label"]==1)]
            one_rule_contrast_neg = one_rule.loc[(one_rule["contrast"]==1)&(one_rule["sentiment_label"]==0)]
            one_rule_contrast_pos_sample = one_rule_contrast_pos.sample(n=1350, random_state=11)
            one_rule_contrast_neg_sample = one_rule_contrast_neg.sample(n=1351, random_state=11)
            one_rule.drop(one_rule_contrast_pos_sample.index, inplace = True)
            one_rule.drop(one_rule_contrast_neg_sample.index, inplace = True)
            results = pd.concat([results.loc[results["rule_label"]==0], one_rule])
            results = results.reset_index(drop=True)
            base_sent_corrects = Corrects_distribution(len(results['sentence'])).model_sentiment_correct_distributions(results)
            accuracy = sum(base_sent_corrects["one_rule"])/len(base_sent_corrects["one_rule"])
            print("\n")
            print("Sentiment accuracy: ", round(accuracy, 3))
        
        if self.config["generate_explanations"] == True:
            print("\nLIME explanations")

            #load trained model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #results to be created after explanation
            explanations = {"sentence":[], 
                            "LIME_explanation":[], 
                            "LIME_explanation_normalised":[]}

            with open("assets/results/"+self.config["asset_name"]+".pickle", 'rb') as handle:
                results = pickle.load(handle)
            results = pd.DataFrame(results)

            test_sentences = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentence']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentence'])
            probabilities = list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)]['sentiment_probability_output']) + list(results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]['sentiment_probability_output'])

            explainer = lime_text.LimeTextExplainer(class_names=["negative_sentiment", "positive_sentiment"], 
                                                    split_expression=" ", 
                                                    random_state=self.config["seed_value"])

            for index, test_datapoint in enumerate(tqdm(test_sentences)):
                probability = [1 - probabilities[index].tolist()[0], probabilities[index].tolist()[0]]
                tokenized_sentence = test_datapoint.split()
                try:
                    exp = explainer.explain_instance(
                                                    test_datapoint, 
                                                    self.prediction, 
                                                    num_features = len(tokenized_sentence), 
                                                    num_samples = self.config["lime_no_of_samples"]
                                                    )
                except:
                    traceback.print_exc()
                    text = test_datapoint
                    explanation = "couldn't process"
                    explanations["sentence"].append(text)
                    explanations["LIME_explanation"].append(explanation)
                    explanations["LIME_explanation_normalised"].append(explanation)
                    continue
                text = []
                explanation = []
                explanation_normalised = []
                for word in test_datapoint.split():
                    for weight in exp.as_list():
                        weight = list(weight)
                        if weight[0]==word:
                            text.append(word)
                            if weight[1] < 0:
                                weight_normalised_negative_class = abs(weight[1])*probability[0]
                                explanation_normalised.append(weight_normalised_negative_class)
                            elif weight[1] > 0:
                                weight_normalised_positive_class = abs(weight[1])*probability[1]
                                explanation_normalised.append(weight_normalised_positive_class)
                            explanation.append(weight[1])
                explanations['sentence'].append(text)
                explanations['LIME_explanation'].append(explanation)
                explanations['LIME_explanation_normalised'].append(explanation_normalised)

                if self.config["generate_explanation_for_one_instance"] == True:
                    print(explanation_normalised)
                    break
            
            if not os.path.exists("assets/lime_explanations/"):
                os.makedirs("assets/lime_explanations/")
            with open("assets/lime_explanations/"+self.config["asset_name"]+".pickle", "wb") as handle:
                pickle.dump(explanations, handle)

            #Check the rule-subset of the results: balancing between contrast and no-contrast tweets of rule-subset
            if self.config["generate_explanation_for_one_instance"] == False:
                results = pd.DataFrame(results)
                one_rule = pd.concat([results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)], 
                                      results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]])
                one_rule = one_rule.reset_index(drop=True)
                one_rule_contrast_pos = one_rule.loc[(one_rule["contrast"]==1)&(one_rule["sentiment_label"]==1)]
                one_rule_contrast_neg = one_rule.loc[(one_rule["contrast"]==1)&(one_rule["sentiment_label"]==0)]
                one_rule_contrast_pos_sample = one_rule_contrast_pos.sample(n=1350, random_state=11)
                one_rule_contrast_neg_sample = one_rule_contrast_neg.sample(n=1351, random_state=11)
                one_rule.drop(one_rule_contrast_pos_sample.index, inplace = True)
                one_rule.drop(one_rule_contrast_neg_sample.index, inplace = True)
                results = pd.concat([results.loc[results["rule_label"]==0], one_rule])
                results = results.reset_index(drop=True)
                explanations = pd.DataFrame(explanations)
                explanations.drop(one_rule_contrast_pos_sample.index, inplace = True)
                explanations.drop(one_rule_contrast_neg_sample.index, inplace = True)
                explanations = explanations.reset_index(drop=True)
                one_rule_results = pd.concat([results.loc[(results["rule_label"]!=0)&(results["contrast"]==1)], 
                                              results.loc[(results["rule_label"]!=0)&(results["contrast"]==0)]])
                percy = calculate_percy(one_rule_results, explanations)
                percy_value = sum(percy["one_rule"])/len(percy["one_rule"])
                print("\n")
                print("PERCY score: ", round(percy_value, 3))
        
        #save the configuration parameters for this run (marks the creation of an asset)
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        