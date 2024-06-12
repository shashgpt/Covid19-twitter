import tensorflow as tf

def gru(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = tf.keras.layers.Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = tf.keras.layers.Embedding(word_vectors.shape[0], 
                                    word_vectors.shape[1], 
                                    embeddings_initializer=tf.keras.initializers.Constant(word_vectors), 
                                    trainable=config["fine_tune_word_embeddings"], 
                                    mask_zero=True, 
                                    name="word2vec")(input_sentence)

    # Classifier Layer
    out = tf.keras.layers.GRU(128, dropout=config["dropout"], name="classifier")(out)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = tf.keras.Model(inputs=[input_sentence], outputs=[out])

    return model