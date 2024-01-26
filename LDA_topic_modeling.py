#Document: some text.
#Corpus: a collection of documents.
#Vector: a mathematically convenient representation of a document.
#Model: an algorithm for transforming vectors from one representation to another.

import pprint
import pandas as pd
from gensim import corpora, models
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from nltk.corpus import stopwords
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt

research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")

# Create a sub-dataset with the first 10 lines
sub_dataset = research.head(10)
# Extract descriptions from the 'description' column of the dataframe
text_corpus = sub_dataset['Abstract'].tolist()
text_corpus = [f'"{doc}"' for doc in text_corpus]

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1 # Increase frecuency Token or decrease depending espected results

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

#%%
# Attempting to directly pass processed_corpus to LdaModel without first converting it to a Bag-of-Words (BoW) format using the doc2bow method for each document

# Create Dictionary for processed corpus
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

# Convert the dictionary to a bag-of-words corpus for reference.
corpus = [dictionary.doc2bow(text) for text in processed_corpus]

""" Train an LDA model using a Gensim corpus """
# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
# Train the LDA model on the BoW corpus.
lda = LdaModel(corpus, num_topics=10)
# Train the LDA model
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
#lda = models.LdaModel(corpus, num_topics=10)

# Create a visualization
vis = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.display(vis)
