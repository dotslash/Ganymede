import pandas
import numpy as np
import random
import ast
import os

data_dir = '~/data/kaggle:jigsaw-unintended-bias-in-toxicity-classification/'
train_file = data_dir + 'train.csv'
test_file = data_dir + 'test.csv'
train_file = os.path.expanduser(train_file)
test_file = os.path.expanduser(test_file)



train_data = pandas.read_csv(train_file)
test_data = pandas.read_csv(test_file)



useless_cols = {'id', 'comment_text','created_date',
                'publication_id', 'parent_id', 'article_id'}
useful_cols = [col for col in train_data.columns if col not in useless_cols]
corr = train_data[useful_cols].corr()
corr["order"] = -np.abs(corr["target"])


# In[13]:


import matplotlib.pyplot as plt
plt.subplots(figsize=(10, 8))
trimmed_corr = corr[["target", "order"]].dropna().sort_values(by="order").head(40)["target"]
trimmed_corr.plot(kind="barh", title = "Correlation with target", figsize = (10,10))


# In[24]:


thead = train_data.head()
thead_text = thead["comment_text"].astype(str)


# In[25]:


from keras.preprocessing import text, sequence


# In[26]:


CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)


# In[27]:


tokenizer.fit_on_texts(list(thead_text))


# In[49]:


thead_text_seq = tokenizer.texts_to_sequences(thead_text)
MAX_SEQUENCE_LENGTH = 100
thead_text_seq = sequence.pad_sequences(thead_text_seq, MAX_SEQUENCE_LENGTH)


# In[50]:


thead_text
thead_text_seq


# In[36]:


tokenizer.word_index


# In[37]:


embedding_file = '~/data/fasttext/crawl-300d-2M.vec'
embedding_file = os.path.expanduser(embedding_file)

embeddings_index = {}
f = open(embedding_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[44]:


EMBEDDING_DIM = len(embeddings_index["hello"])
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector




# In[51]:


from keras.layers import Embedding
from keras.initializers import Constant
embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[ ]:





