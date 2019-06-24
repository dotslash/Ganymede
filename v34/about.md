[script version link](https://www.kaggle.com/yesteapea/kernel7172c9dda3?scriptVersionId=16187082)

Flags
```python3
LSTM_UNITS: int = 128
DENSE_HIDDEN_UNITS: int = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH: int = 200
USE_CATEGORY_COLS: bool = True
TESTING_MODE: bool = False
ENABLE_TEXT_PROCESSING: bool = True
BATCH_SIZE: int = 1024
NUM_EPOCHS: int = 20
TOKENIZER_NUM_WORDS: int = 50000
```

Uses sample weights
```python3
# Focus more on the rows that have identity columns.
sample_weights += train_id_columns_sum
# Focus more on the false negatives
sample_weights += (bool_target * inv_train_id_columns_sum)
# Focus a lot more on the false positives
sample_weights += ((inv_bool_target * train_id_columns_sum) * 5)
sample_weights /= sample_weights.mean()
```

