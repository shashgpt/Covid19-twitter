import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import os
import numpy as np
import pandas as pd
import pickle
from lime import lime_text
import traceback
from tqdm import tqdm
from transformers import TFAutoModel, AutoConfig, AutoTokenizer

from scripts.training.additional_validation_sets import AdditionalValidationSets

MODEL = "distilbert/distilbert-base-uncased"
EMBEDDING_SIZE_KEY="dim"

class distilbert_cnn(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.configuration = AutoConfig.from_pretrained(MODEL).to_dict()
        self.encoder = TFAutoModel.from_pretrained(MODEL)
        for layer in self.encoder.layers:
            layer.trainable = config["fine_tune_word_embeddings"]

        self.conv_1 = layers.Conv2D(filters = self.config["hidden_units"], 
                                    kernel_size = (3, self.configuration[EMBEDDING_SIZE_KEY]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_1")
        self.conv_2 = layers.Conv2D(filters = self.config["hidden_units"], 
                                    kernel_size = (4, self.configuration[EMBEDDING_SIZE_KEY]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_2")
        self.conv_3 = layers.Conv2D(filters = self.config["hidden_units"], 
                                    kernel_size = (5, self.configuration[EMBEDDING_SIZE_KEY]),
                                    strides = 1,
                                    dilation_rate = 1,
                                    padding = "valid",
                                    data_format = "channels_first",
                                    name = "conv2D_3")
        self.relu = layers.ReLU()
        self.max_pool = layers.GlobalMaxPooling1D(data_format="channels_first",
                                                  name="maxpool1D")
        self.concat = layers.Concatenate(axis=1, name="concatenate")
        self.dropout = layers.Dropout(rate=config["dropout"], 
                                      seed=config["seed_value"], 
                                      name="dropout")
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")
    
    def call(self, input_ids, training, **kwargs):

        #encoder output
        word_embeddings = self.encoder(input_ids).last_hidden_state
        word_embeddings_reshaped = tf.keras.backend.expand_dims(word_embeddings, axis=1) # batch_size x 1 x sent_len x embedding_dim
        
        output_1 = self.conv_1(word_embeddings_reshaped)
        output_1 = tf.keras.backend.squeeze(output_1, axis=3)
        output_1 = self.relu(output_1)
        output_1 = self.max_pool(output_1)

        output_2 = self.conv_2(word_embeddings_reshaped)
        output_2 = tf.keras.backend.squeeze(output_2, axis=3)
        output_2 = self.relu(output_2)
        output_2 = self.max_pool(output_2)

        output_3 = self.conv_3(word_embeddings_reshaped)
        output_3 = tf.keras.backend.squeeze(output_3, axis=3)
        output_3 = self.relu(output_3)
        output_3 = self.max_pool(output_3)

        output = self.concat([output_1, output_2, output_3])
        output = self.dropout(output, training=training)

        out = self.out(output)
        return out

    def build_model(self, input_shape):
        input_data = tf.keras.layers.Input(shape=(input_shape,), dtype="int64")
        model = tf.keras.Model(inputs=input_data, outputs=self.call(input_data, training=False))
        model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=self.config["learning_rate"]), 
                                                        loss=['binary_crossentropy'], 
                                                        metrics=['accuracy'])
        model.summary()
        return model

class train_distilbert_cnn(object):
    def __init__(self, config) -> None:
        self.config = config

    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset using bert tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
        pred_prob_1 = self.model.predict(x)
        pred_prob_0 = 1 - pred_prob_1
        prob = np.concatenate((pred_prob_0, pred_prob_1), axis=1)
        return prob
    
    def train_model(self, train_dataset, val_datasets, test_datasets):

        #Make paths
        if not os.path.exists("assets/training_history/"):
            os.makedirs("assets/training_history/")

        #Create train, val, and test datasets
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

        #Create additional validation datasets
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentences = self.pad(sentences, maxlen)
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        #Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # 1. Calculate val_loss_1 
                                                                    min_delta = 0,      # 2. Check val_losses for next 10 epochs 
                                                                    patience=10,        # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                    verbose=0,          # 4. Get the trained weights corresponding to val_loss_1
                                                                    mode="min",
                                                                    baseline=None, 
                                                                    restore_best_weights=True)
        my_callbacks = [
                        early_stopping_callback, 
                        AdditionalValidationSets(additional_validation_datasets, self.config)
                       ]

        #model compilation and summarization
        model = distilbert_cnn(self.config).build_model(input_shape = train_dataset[0].shape[1])
        self.model = model

        # Train the model
        if self.config["train_model"] == True:
            self.model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    batch_size=self.config["mini_batch_size"], 
                    epochs=self.config["train_epochs"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks)

            # Save trained model
            if not os.path.exists("assets/trained_models/"):
                os.makedirs("assets/trained_models/")
            self.model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")
        
        if self.config["evaluate_model"] == True:

            #Load model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #Results to be created after evaluation
            results = test_datasets["test_dataset"].copy()

            #Evaluation and predictions
            evaluations = self.model.evaluate(x=test_dataset[0], y=test_dataset[1])
            print("test loss, test acc:", evaluations)
            predictions = self.model.predict(x=test_dataset[0])
            print(len(predictions))

            #Create results
            results['sentiment_probability_output'] = []
            results['sentiment_prediction_output'] = []
            for prediction in predictions:
                results['sentiment_probability_output'].append(prediction)
                prediction = np.rint(prediction)
                results['sentiment_prediction_output'].append(prediction[0])

            #Save the results
            if not os.path.exists("assets/results/"):
                os.makedirs("assets/results/")
            with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
                pickle.dump(results, handle)

        if self.config["generate_explanation"] == True:
            print("\nLIME explanations")

            #Load trained model
            self.model.load_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

            #Results to be created after explanation
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
        
        #save the configuration parameters for this run (marks the creation of an asset)
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)