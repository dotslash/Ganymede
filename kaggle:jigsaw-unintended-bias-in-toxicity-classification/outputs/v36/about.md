[script version link](https://www.kaggle.com/yesteapea/kernel7172c9dda3?scriptVersionId=16221816)

Flags
```python3
LSTM_UNITS: int = 128
DENSE_HIDDEN_UNITS: int = 4 * LSTM_UNITS
MAX_SEQUENCE_LENGTH: int = 200
USE_CATEGORY_COLS: bool = True
TESTING_MODE: bool = False
ENABLE_TEXT_PROCESSING: bool = False
BATCH_SIZE: int = 1024
NUM_EPOCHS: int = 3
TOKENIZER_NUM_WORDS: int = 50000
NUM_MODELS:int = 2
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


Model fitting
```python3
for model_idx in range(NUM_MODELS):
  model = build_model(embedding_matrix, y_other_train.shape[-1])
  for global_epoch in range(NUM_EPOCHS):
    weights.append(2 ** global_epoch)
    model.fit(
        x_train, [y_train, y_other_train],
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        epochs=1,
        # One set of sample_weights for each output
        sample_weight=[sample_weights.values,
                       sample_weights.values],
        callbacks=[learning_rate_ctrl])
    logger.log('Trained model.')
    checkpoint_predictions.append(
    model.predict(x_test, batch_size=2048)[0])
y_test = np.average(checkpoint_predictions, weights=weights, axis=0)
```