#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from copy import deepcopy


#Import Dataset
df = pd.read_csv('commit_augmented.csv', sep= ',', encoding='utf-8')
df.head()

# Step 1: Load GloVe word embeddings
embedding_dim = 300  # Adjust the embedding dimension based on your GloVe model
glove_path = './glove.840B.300d.txt'  # Provide the path to your GloVe file

# Load GloVe embeddings into a dictionary
embedding_index = {}
with open(glove_path, 'r', errors ='ignore', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embedding_index[word] = coefs
        

df.dropna(inplace=True)
# Prepare text data and labels
df['text'].dropna(inplace=True)
texts = df['text'].tolist()
#df = df[df["classification"]!="non_debt"]
df = df[df["classification"]!="architecture_debt"]
df = df[df["classification"]!="build_debt"]
df = df[df["classification"]!="defect_debt"]

#df.classification.value_counts()

#defining a list of the classes names
labels = ["non_debt", "satd"]

#labels = df['classification'].replace({'satd':1.0, 'documentation_debt':1.0, 'non_debt':0.0}, inplace=True)
df['classification'].replace({'non_debt':0.0, 'design_debt':1.0, 'code_debt':1.0, 'requirement_debt':1.0, 'test_debt':1.0, 'documentation_debt':1.0, 'code-design_debt':1.0}, inplace=True)

labels = df['classification'].tolist()
df.classification.value_counts()


# Tokenize and pad the text data
max_sequence_length = 256  # Adjust the sequence length as needed
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Create an embedding matrix
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        

# prepare the labels
X = padded_sequences
y = df.classification.values
y = np.asarray(y).astype("float64")


print('splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=1)

print(len(X_train))
print(len(X_val))
print(len(X_test))

y_train_enc = pd.get_dummies(y_train).to_numpy()
y_val_enc = pd.get_dummies(y_val).to_numpy()
y_test_enc = pd.get_dummies(y_test).to_numpy()


#Build the BiLSTM model
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))

model.add(Bidirectional(LSTM(128, return_sequences=True)))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# Step 7: Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=5e-5), metrics=['accuracy']) #1e-3 5e-5 0.005
model.summary()


# EarlyStopping and ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
mc = ModelCheckpoint('./model_bilstm_augmented_commit.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)



with tf.device('/GPU:0'):
    history_embedding = model.fit(X_train, y_train_enc, 
                                    epochs = 40, batch_size = 16, 
                                    validation_data=(X_val, y_val_enc),
                                    verbose = 1, callbacks= [es, mc]  )
    

plt.plot(history_embedding.history['loss'],c='b',label='train')
plt.plot(history_embedding.history['val_loss'],c='r',label='validation')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


plt.plot(history_embedding.history['accuracy'],c='b',label='train')
plt.plot(history_embedding.history['val_accuracy'],c='r',label='validation')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('val accuracy')
plt.show()


y_pred = np.argmax(model.predict(X_test), axis  =  1)
y_true = np.argmax(y_test_enc, axis = 1)
print(classification_report(y_pred, y_true, target_names=['non_debt','satd'], digits=3))
    
    