import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup as bs
import requests

url = 'https://guides.loc.gov/federalist-papers/full-text'

html = requests.get(url).text
soup = bs(html, 'html.parser')

# convert the html table to a pandas dataframe
table = soup.find('table')

# List of Federalist Papers
meta_df = pd.read_html(StringIO(str(table)), parse_dates=True)[0]

# disputed papers
disputed_papers = meta_df[meta_df['Author'] == 'Hamilton or Madison']
contested_authorship = disputed_papers['No.'].values
print(f'Contested authorship of papers: {contested_authorship}')
print(f'Total number of disputed papers: {len(disputed_papers)}')

corpus = pd.read_pickle('fp_corpus.pkl')
corpus.head()

# drop the duplicates and keep target and paper_id
authorship = corpus.drop_duplicates(subset=['paper_id', 'target'])
authorship.target.value_counts()

# YOUR CODE HERE - CREATE A DATASET FOR THE CLASSIFICATION TASK OF AUTHORSHIP ATTRIBUTION

# since we want to classify the disputed papers,
# let's remove them from the dataset into their own dataset.

disputed_authors = corpus[corpus['target'] == 'dispt']

# It stands to reason that the disputed papers could be co-authored, so
# let's remove them from the dataset and use them later. Our first task
# is to classify the disputed papers wrt to the question: For any given
# disputed paper, is it more likely to be authored by Hamilton or Madison?

coauthored = corpus[corpus['target'] == 'HM']

# remove the coauthored papers from the dataset
corpus = corpus[corpus['target'] != 'HM']

# since the disputed authors are either Hamilton or Madison, we can
# remove Jay from the dataset. We will hold Jay off for now, but we might
# want to experiment with Jay later.

# create a mask to filter out Jay
jay = corpus[corpus['target'] == 'Jay']

# use the mask to remove jay from the dataset
corpus = corpus[corpus['target'] != 'Jay']

# remove disputed papers from the corpus
corpus = corpus[corpus['target'] != 'dispt']

# plot the sentence length distribution
plt.figure(figsize=(10, 6))
sns.histplot(corpus['sentence_length'], bins=50)
plt.title('Sentence Length Distribution')
plt.show()

# YOUR CODE HERE - Visualize the distribution of the sentence lengths (what is the quartile distribution of the sentence lengths?)
import numpy as np

# get the 95th percentile of the sentence length
max_len = np.percentile(corpus['sentence_length'], 95)
max_len

from sklearn.model_selection import train_test_split

X_train, x_val, y_train, y_val = train_test_split(corpus['sentences'],
                                                  corpus['target'],
                                                  test_size=0.2,
                                                  stratify=corpus['target'],
                                                  random_state=42
                                )

print(f'Training samples: {len(X_train)}')
print(f'Validation samples: {len(x_val)}')

import random

# sample a random sentence from the training set
random_idx = random.randint(0, len(X_train))
print(f'Random sentence: {X_train.iloc[random_idx]}')
print(f'Author: {y_train.iloc[random_idx]}')

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

y_train_ohe = ohe.fit_transform(y_train.values.reshape(-1, 1))
y_val_ohe = ohe.transform(y_val.values.reshape(-1, 1))

print(f'One hot encoded training labels shape: {y_train_ohe.shape}')
print(f'One hot encoded validation labels shape: {y_val_ohe.shape}')

## One hot encoded training labels shape: (3904, 2)
## One hot encoded validation labels shape: (977, 2)

y_train_ohe[:5], y_train[:5]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_le = label_encoder.fit_transform(y_train)
y_val_le = label_encoder.transform(y_val)

print(f'Label encoded training labels shape: {y_train_le.shape}')
print(f'Label encoded validation labels shape: {y_val_le.shape}')

for i in range(5):
    print(f'Original label: {y_train.iloc[i]} - Label encoded: {y_train_le[i]}')

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
num_classes, class_names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear', C=1.0, random_state=42))
])

# fit the model
text_clf.fit(X_train, y_train_le)

text_clf.score(x_val, y_val_le)

# let's use grid search to find the best hyperparameters for the model
from sklearn.model_selection import GridSearchCV

parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__min_df': [1, 5, 10],
    'clf__C': [0.1, 1.0, 15.0]
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train_le)

print(f'{gs_clf.best_params_=}, {gs_clf.best_score_=}')

gs_clf.score(x_val, y_val_le)

# save the best model
import joblib

joblib.dump(gs_clf, 'authorship_attribution_model.pkl')

# load the model
model = joblib.load('authorship_attribution_model.pkl')

# Run inference on the disputed papers
disputed_predictions = model.predict(disputed_authors['sentences'])

# for a disputed paper plot the predicted probabilities over the classes and sentence order
import matplotlib.pyplot as plt
import seaborn as sns

# add the predicted probabilities to the disputed authors dataframe using the labels
disputed_authors['predicted'] = label_encoder.inverse_transform(disputed_predictions)

# let's plot the authorship classification for the disputed paper 49
dp_49 = disputed_authors[disputed_authors['paper_id'] == '49']

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

max_vocab_size = 1000  
max_sequence_length = 100  
output_sequence_length = 20  
pad_to_max_tokens = True  

text_vectorizer = TextVectorization(
    max_tokens=max_vocab_size,  
    output_mode='int',  
    output_sequence_length=output_sequence_length,  
    pad_to_max_tokens=pad_to_max_tokens  
)

# Randomly visualize some of your vectorized textual data
import random

text_vectorizer.adapt(X_train.values)

example_sent = random.choice(X_train.values) # change var name if you need to
print(f'Original text:\n{example_sent}')
print(f'\nVectorized text:\n{text_vectorizer([example_sent])}')
print('Length of vector:', len(text_vectorizer([example_sent]).numpy()[0]))

# examine the vocabulary
vocab = text_vectorizer.get_vocabulary()
print(f'Number of words in the vocabulary: {len(vocab)}')
print(f'Most common words in the vocabulary: {vocab[:5]}')
print(f'Least common words in the vocabulary: {vocab[-5:]}')

# Adapt the text vectorizer to the training data
text_vectorizer.adapt(X_train)

# Get the vocabulary size from the text_vectorizer
vocab_size = len(text_vectorizer.get_vocabulary())

# Define the Embedding layer with the correct input_dim
token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim=128,
                                            mask_zero=True,
                                            name='token_embedding')

print(f'Sentence before vectorization: {example_sent}')
vectorized_sent = text_vectorizer(example_sent)
print(f'Sentence after vectorization: {vectorized_sent}')
embedded_sent = token_embedding(vectorized_sent)
print(f'Sentence after embedding: {embedded_sent}')

X_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_le))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val_le))

# print the first 5 samples
for sample in X_dataset.take(5):
    sentence, label = sample
    print(f'Sentence: {sentence} - Label: {label}')

BATCH_SIZE = 32

train_dataset = X_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = 'categorical_crossentropy'
epochs = 8