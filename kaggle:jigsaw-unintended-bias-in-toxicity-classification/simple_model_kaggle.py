import pandas
import numpy as np
from keras.preprocessing import text, sequence
from keras.layers import Embedding
from keras.initializers import Constant
import os
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, add, concatenate
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM
import datetime

def pretty_time_delta(delta: datetime.timedelta):
    seconds = delta.total_seconds()
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '{}{}d{}h{}m{}s'.format(sign_string, days, hours, minutes, seconds)
    else:
        return '{}{}h{}m{}s'.format(sign_string, hours, minutes, seconds)


class Logger:
    def __init__(self):
        self.start = datetime.datetime.now()
    def Log(self, message):
        time_taken = pretty_time_delta(
            datetime.datetime.now() - self.start)
        print('{}: {}'.format(time_taken, message))

KAGGLE_MODE = True
MAX_TRAIN_SAMPLES = 10**4

logger = Logger()
logger.Log("Started.")
train_file = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_file = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
embedding_file = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
if not KAGGLE_MODE:
    data_dir = '~/data/kaggle:jigsaw-unintended-bias-in-toxicity-classification/'
    train_file = data_dir + 'train.csv'
    test_file = data_dir + 'test.csv'
    embedding_file = '~/data/fasttext/crawl-300d-2M.vec'
    
    embedding_file = os.path.expanduser(embedding_file)
    train_file = os.path.expanduser(train_file)
    test_file = os.path.expanduser(test_file)

train_data = pandas.read_csv(train_file)
train_data = train_data[["comment_text", "target"]]
np.random.shuffle(train_data.values)
if train_data.size > MAX_TRAIN_SAMPLES:
    train_data = train_data.head(MAX_TRAIN_SAMPLES)
test_data = pandas.read_csv(test_file)
logger.Log("Loaded train and test data.")


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train_data["comment_text"]) + list(test_data["comment_text"]))
logger.Log("Fit text tokens.")

MAX_SEQUENCE_LENGTH = 200
train_seq = tokenizer.texts_to_sequences(list(train_data["comment_text"]))
train_seq = sequence.pad_sequences(train_seq, MAX_SEQUENCE_LENGTH)
test_seq = tokenizer.texts_to_sequences(list(test_data["comment_text"]))
test_seq = sequence.pad_sequences(test_seq, MAX_SEQUENCE_LENGTH)
logger.Log("Converted tokens to sequences.")

split_ind = int(len(train_seq)*0.7)
X_train, Y_train = train_seq[:split_ind], train_data["target"][:split_ind]
X_val, Y_val = train_seq[split_ind:], train_data["target"][split_ind:]
X_test = test_seq
logger.Log("Prepared and train, validation and test sets.")


embeddings_index = {}
f = open(embedding_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
logger.Log('Loaded embedding file')
logger.Log('Found %s word vectors.' % len(embeddings_index))


EMBEDDING_DIM = len(embeddings_index["hello"])
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

logger.Log('Created embedding matrix.')


LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

def build_model(embedding_matrix):
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], 
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    result = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, output=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

model = build_model(embedding_matrix)
logger.Log('Training model.')
model.fit(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    batch_size=128,
    epochs=1, verbose=2)
logger.Log('Trained model.')
Y_test = model.predict(X_test, batch_size=2048)
logger.Log('Predicted test set.')

submission = pandas.DataFrame.from_dict({
    'id': test_data.id,
    'prediction': Y_test.flatten()
})
submission.to_csv('submission.csv', index=False)
logger.Log('Done.')
