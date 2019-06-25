import datetime
from typing import List, Tuple

import numpy as np
import pandas
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.initializers import Constant
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, CuDNNLSTM
from keras.layers import Input, Dense, SpatialDropout1D
from keras.layers import add, concatenate
from keras.models import Model
from keras.preprocessing import text, sequence

# Flags.
LSTM_UNITS: int = 128
DENSE_HIDDEN_UNITS: int = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH: int = 200
TESTING_MODE: bool = False
ENABLE_TEXT_PROCESSING: bool = False
BATCH_SIZE: int = 1024
NUM_EPOCHS: int = 4
TOKENIZER_NUM_WORDS: int = 50000
NUM_MODELS: int = 2

# Facts.
CATEGORY_COLS: List[str] = ['severe_toxicity', 'obscene', 'identity_attack',
                            'insult', 'threat']
IDENTITY_COLS: List[str] = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# TODO(dotslash): Make the file paths work both in kaggle and locally.
TRAIN_FILE: str = \
    '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'
TEST_FILE: str = \
    '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
EMBEDDING_FILES: List[str] = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt']


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

    def log(self, message: str) -> None:
        now = datetime.datetime.now()
        time_taken = pretty_time_delta(now - self.start)
        print('{} delta-{}: {}'.format(now, time_taken, message))


logger = Logger()
logger.log('Started.')


class EmbeddingStore:
    def __init__(self, embedding_file: str):
        f = open(embedding_file)
        logger.log('Loading embedding file:{}'.format(embedding_file))
        self.dict = dict()
        for line in f:
            if TESTING_MODE and len(self.dict) > 100000:
                # 100k words are enough if we are in test mode
                break
            values = line.strip().split(' ')
            word = values[0]
            try:
                coeffs = np.asarray(values[1:], dtype='float32')
                self.vector_length = len(coeffs)
                self.dict[word] = coeffs
            except Exception:
                logger.log('Failed parsing embedding for "{}"'.format(word))
        f.close()
        logger.log('Loaded embedding file: {}'.format(embedding_file))
        logger.log('Found %s word vectors.' % len(self.dict))

    def embedding(self, word: str) -> np.array:
        return self.dict.get(word, np.zeros(self.vector_length))


def get_top_words(tokenizer: text.Tokenizer):
    ret = [(v, k) for k, v in tokenizer.index_word.items()]
    return ret[:tokenizer.num_words]


def binary_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    import keras.backend as kb
    return kb.mean(kb.equal(kb.round(y_true), kb.round(y_pred)))


def build_model(embedding_matrix: np.array, num_other_results: int):
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
    other_results = Dense(num_other_results, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=[result, other_results])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['acc', binary_accuracy])
    return model


def print_diff(s1: pandas.Series, s2: pandas.Series) -> None:
    diff: pandas.Series = (s1 == s2)
    logger.log('diff')
    print(diff.value_counts())


# TODO(dotslash): Create a container type for return value of this
#                 function.
def load_train_data() -> Tuple[np.array, np.array,
                               np.array, np.array,
                               pandas.DataFrame, pandas.DataFrame,
                               text.Tokenizer]:
    # Load Training and Testing data.
    train_data: pandas.DataFrame = pandas.read_csv(TRAIN_FILE)
    test_data: pandas.DataFrame = pandas.read_csv(TEST_FILE)
    np.random.shuffle(train_data.values)
    logger.log('Loaded train and test data.')

    if TESTING_MODE:
        train_data = train_data.head(10000)
        test_data = test_data.head(10000)

    # Trim the train data and keep only the useful columns.
    useful_cols: List[str] = \
        ['id', 'comment_text', 'target'] + CATEGORY_COLS + IDENTITY_COLS
    train_data: pandas.DataFrame = train_data[useful_cols]
    print('Sample training data\n' + train_data.head().to_string())
    print('Sample test data\n' + test_data.head().to_string())

    # Create a tokenizer based on train and test data.
    tokenizer: text.Tokenizer = text.Tokenizer(num_words=TOKENIZER_NUM_WORDS)
    tokenizer.fit_on_texts(list(train_data['comment_text']) + \
                           list(test_data['comment_text']))
    logger.log('Fit text tokens.')

    # Prepare X, Y for training and testing.
    # We will convert the text to a sequence using the tokenizer.
    train_seq = tokenizer.texts_to_sequences(list(train_data['comment_text']))
    train_seq = sequence.pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH)
    test_seq = tokenizer.texts_to_sequences(list(test_data['comment_text']))
    test_seq = sequence.pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LENGTH)
    logger.log('Converted tokens to sequences.')

    x_train, y_train, y_other_train = \
        train_seq, train_data['target'], train_data[CATEGORY_COLS]
    x_test = test_seq
    logger.log('Prepared and train, validation and test sets.')
    return x_train, y_train, y_other_train, x_test, train_data, test_data, tokenizer


def construct_embedding_matrix(tokenizer: text.Tokenizer) -> np.array:
    # Load embeddings from disk.
    embeddings = [EmbeddingStore(embedding_file)
                  for embedding_file in EMBEDDING_FILES]
    # Construct a embedding matrix used for Embedding layer.
    embedding_dim = sum(embedding.vector_length for embedding in embeddings)
    tokenizer_words = get_top_words(tokenizer)
    embedding_matrix = np.zeros((len(tokenizer_words) + 1, embedding_dim))
    for word, ind in tokenizer_words:
        embedding_matrix[ind] = np.concatenate(
            [embedding.embedding(word) for embedding in embeddings])
    logger.log('Created embedding matrix.')
    return embedding_matrix


def main():
    x_train, y_train, y_other_train, x_test, train_data, \
        test_data, tokenizer = load_train_data()
    embedding_matrix = construct_embedding_matrix(tokenizer)
    sample_weights: pandas.Series = pandas.Series(
        data=np.ones(len(x_train), dtype=np.float32))

    for column in IDENTITY_COLS:
        train_data[column] = np.where(train_data[column] >= 0.5, True, False)
    bool_target: pandas.Series = pandas.Series(
        data=np.where(train_data['target'] > 0.5, True, False))
    inv_bool_target: pandas.Series = ~bool_target
    train_id_columns_sum: pandas.Series = train_data[IDENTITY_COLS].sum(axis=1)
    inv_train_id_columns_sum: pandas.Series = (~train_data[IDENTITY_COLS]).sum(
        axis=1)
    # Focus more on the rows that have identity columns.
    sample_weights += train_id_columns_sum
    # Focus more on the false negatives
    sample_weights += (bool_target * inv_train_id_columns_sum)
    # Focus a lot more on the false positives
    sample_weights += ((inv_bool_target * train_id_columns_sum) * 5)
    sample_weights /= sample_weights.mean()

    # Fit the model.
    logger.log('Training model.')
    checkpoint_predictions = []
    weights = []
    for model_idx in range(NUM_MODELS):
        model = build_model(embedding_matrix, y_other_train.shape[-1])
        for global_epoch in range(NUM_EPOCHS):
            model.fit(
                x_train, [y_train, y_other_train],
                validation_split=0.1,
                batch_size=BATCH_SIZE,
                epochs=1,
                # One set of sample_weights for each output
                sample_weight=[sample_weights.values, sample_weights.values],
                # TODO(dotslash): How does this help?
                callbacks=[LearningRateScheduler(
                    lambda _: (0.55 ** global_epoch) / 1000.0, verbose=1)])
            logger.log('Trained model: {}.'.format(model_idx))
            weights.append(2 ** global_epoch)
            checkpoint_predictions.append(
                model.predict(x_test, batch_size=2048)[0])
    y_test = np.average(checkpoint_predictions, weights=weights, axis=0)
    logger.log('Predicted test set.')
    submission = pandas.DataFrame.from_dict({
        'id': test_data.id,
        'prediction': y_test.flatten()
    })
    submission.to_csv('submission.csv', index=False)
    logger.log('Done.')


if __name__ == '__main__':
    main()
