import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
#data = pd.read_csv('./combined_data.csv')
data = pd.read_csv('./processed_twitter_data.csv');
#text = data["text"];
#sentiment = data["sentiment"];
sentiment = data.iloc[:, 0];
text = data.iloc[:, 5];
def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens


X_train, X_test , y_train, y_test = train_test_split(text, sentiment, test_size = 0.20)
vocab_size = 10000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
print(X_train.shape)
print(X_train);
print(X_test.shape);
print(y_train.shape)
print(y_test.shape);


max_length = 50
padding_type='post'
truncation_type='post'
X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, padding=padding_type, truncating=truncation_type)
X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type,truncating=truncation_type)

print(X_train_padded.shape)
print(X_test_padded.shape);
print(y_train.shape)
print(y_test.shape);

embeddings_index = dict();
#f = open('clean_pretrained_embeddings.txt')
f = open('gender_debiased_embeddings.embed')
#dimension size is 50
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, max_length))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(input_dim=len(word_index) + 1,
                    output_dim=max_length,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False)
model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(150, return_sequences=True)),
    Bidirectional(LSTM(150)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

log_folder = 'logs'
callbacks = [
            EarlyStopping(patience = 10),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 600
print("so far so good")
print(X_train_padded.shape)
print(X_test_padded.shape);
print(y_train.shape)
print(y_test.shape);
history = model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test),callbacks=callbacks)
loss, accuracy = model.evaluate(X_test_padded,y_test)
print('Test accuracy :', accuracy)
