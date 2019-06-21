import pandas
import numpy as np
from keras.preprocessing import text, sequence
from keras.layers import Embedding
from keras.initializers import Constant
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, add, concatenate
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNLSTM

category_cols = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 
                 'threat']

identity_cols = [
  'asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian',
  'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
  'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim',
  'other_disability', 'other_gender', 'other_race_or_ethnicity',
  'other_religion', 'other_sexual_orientation', 'physical_disability', 
  'psychiatric_or_mental_illness', 'transgender', 'white']

rating_cols = ['rating', 'funny', 'wow', 'sad', 'likes', 'disagree',
               'sexual_explicit']

annotator_cols = ['identity_annotator_count', 'toxicity_annotator_count']

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH = 200
USE_CATEGORY_COLS = True

import datetime
def pretty_time_delta(delta: datetime.timedelta):
    seconds = delta.total_seconds()
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    else:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)

class Logger:
    def __init__(self):
        self.start = datetime.datetime.now()
    def Log(self, message):
        now = datetime.datetime.now()
        time_taken = pretty_time_delta(now - self.start)
        print('{} delta-{}: {}'.format(now, time_taken, message))


class EmbeddingStore:
    def __init__(self, embedding_file):
        f = open(embedding_file)
        logger.Log('Loading embedding file:{}'.format(embedding_file))
        self.dict = dict()
        for line in f:
            values = line.strip().split(' ')
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                self.vector_length = len(coefs)
                self.dict[word] = coefs
            except Exception:
                logger.Log('Failed parsing embedding for "{}"'.format(word))
        f.close()
        logger.Log('Loaded embedding file: {}'.format(embedding_file))
        logger.Log('Found %s word vectors.' % len(self.dict))
    def Embedding(self, word):
        return self.dict.get(word, np.zeros(self.vector_length))


def GetTopWords(tokenizer):
    ret = [(v,k) for k,v in tokenizer.index_word.items()]
    return ret[:tokenizer.num_words]

def build_model(embedding_matrix, num_other_results):
    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], 
                  embeddings_initializer=Constant(embedding_matrix),
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x)])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    x = add([x, Dense(DENSE_HIDDEN_UNITS, activation='relu')(x)])
    result = Dense(1, activation='sigmoid')(x)
    if USE_CATEGORY_COLS:
        other_results = Dense(num_other_results, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=[result, other_results])
    else:
        model = Model(inputs=inp, outputs=[result])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


logger = Logger()
logger.Log("Started.")
train_file = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
test_file = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
embedding_files = ['../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec', 
                   '../input/glove840b300dtxt/glove.840B.300d.txt']

# Load Training and Testing data.
train_data = pandas.read_csv(train_file)
test_data = pandas.read_csv(test_file)
np.random.shuffle(train_data.values)
# train_data = train_data.head(1000)
# test_data = test_data.head(1000)
logger.Log("Loaded train and test data.")

# Trim the train data and keep only the useful columns.
for cat_col in category_cols:
    train_data[cat_col] = train_data[cat_col] > 0.5
useful_cols = ['id', 'comment_text', 'target'] + category_cols
train_data = train_data[useful_cols]
print("Sample training data\n" + train_data.head().to_string())
print("Sample test data\n" + test_data.head().to_string())

# Create a tokenizer based on train and test data.
tokenizer = text.Tokenizer(num_words=50000)
tokenizer.fit_on_texts(list(train_data["comment_text"]) + \
                       list(test_data["comment_text"]))
logger.Log("Fit text tokens.")

# Prepare X, Y for training and testing.
# We will convert the text to a sequence using the tokenizer.
train_seq = tokenizer.texts_to_sequences(list(train_data["comment_text"]))
train_seq = sequence.pad_sequences(train_seq, MAX_SEQUENCE_LENGTH)
test_seq = tokenizer.texts_to_sequences(list(test_data["comment_text"]))
test_seq = sequence.pad_sequences(test_seq, MAX_SEQUENCE_LENGTH)
logger.Log("Converted tokens to sequences.")

X_train, Y_train, Y_other_train = \
    train_seq, train_data["target"], train_data[category_cols]
X_test = test_seq
logger.Log("Prepared and train, validation and test sets.")

# Load embeddings from disk.
embeddings = [EmbeddingStore(embedding_file) \
              for embedding_file in embedding_files]
# Construct a embedding metrix used for Embedding layer.
EMBEDDING_DIM = sum(embedding.vector_length for embedding in embeddings)
tokenizer_words = GetTopWords(tokenizer)
embedding_matrix = np.zeros((len(tokenizer_words) + 1, EMBEDDING_DIM))
for word, ind in tokenizer_words:
    embedding_matrix[ind] = np.concatenate(
            [embedding.Embedding(word) for embedding in embeddings])
logger.Log('Created embedding matrix.')

# Build Model.
model = build_model(embedding_matrix, len(category_cols))
# Fit the model.
logger.Log('Training model.')
if USE_CATEGORY_COLS:
    model.fit(
        X_train, [Y_train, Y_other_train],
        validation_split=0.1,
        batch_size=128,
        epochs=1, verbose=2)
else:
    model.fit(
        X_train, [Y_train],
        validation_split=0.1,
        batch_size=128,
        epochs=1, verbose=2)
logger.Log('Trained model.')

# Predict and prepare submission.
if USE_CATEGORY_COLS:
    Y_test = model.predict(X_test, batch_size=2048)[0]
else:
    Y_test = model.predict(X_test, batch_size=2048)
logger.Log('Predicted test set.')
submission = pandas.DataFrame.from_dict({
    'id': test_data.id,
    'prediction': Y_test.flatten()
})
submission.to_csv('submission.csv', index=False)
logger.Log('Done.')