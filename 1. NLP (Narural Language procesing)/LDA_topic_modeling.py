#%%
"""LDA Model: Prepare requirements described on README.md file, Topic Modeling Gensim file and official documentation disposed throught this project"""
LDA_documentation = 'https://radimrehurek.com/gensim/models/ldamodel.html'

Key_concepts = [
#Document: some text.
#Corpus: a collection of documents.
#Vector: a mathematically convenient representation of a document.
#Model: an algorithm for transforming vectors from one representation to another.
]

import pprint
from collections import defaultdict
import pandas as pd
from gensim import corpora, models
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import spacy
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

""""Data: AI Business related research"""

research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")
# Create a sub-dataset with the first 10 lines
sub_dataset = research
# Extract descriptions from the 'description' column of the dataframe
text_corpus = sub_dataset['Abstract'].tolist()
text_corpus = [f'"{doc}"' for doc in text_corpus]

#%%
""" Clean words matrix """

def get_lemma(token):
    return WordNetLemmatizer().lemmatize(token)
# Create a set of stopwords (set of frequent words):
#stop_words = set(stopwords.words('english'))  # nltk stopwords in english for English 
stop_words = set(nlp.Defaults.stop_words)
# Add manually another stop word to reduce model noise
new_stop_words = {'&','⊆','=','≥','•','de','springer-verlag','et','results','power','±','new','compared','data','smart','la','archaeological','abstract','om','“no','available','set','problem','teatures','switzerland','berlin','ag.','≤','"we','"in','"this','ag."','ieee."','μ','-','z','w','author(s)."','siven','class','rights','general','cows','milk','relevant','reserved."','time','•','native', '1998', 'u', 'x-ray', 'v','pattern','constraint','classiters','vve I','problems','consider','propositional','present','space','large','springer','prove','intake','la','©','found','ieee.','problem','features','given','p','(p)','knowledge','results','si','+','k','n','b','m','f','c','x','d','"[no','available]"','use','field','"The','provide','based','paper','propose','decision','2021','process','methods','paper,','however','number','studies','study','<','conclusion:','2','1','significant','included','total','(n','des','les','genetic','à','en','le','une','qui','du','associated','literature','review',',','intelligence','artificial', 'approach','proposed','intelligence','accuracy','parameters','group','methods:','results:','(p','low', 'different', 'higher', 'analysis', 'published', 'articles','coding', 'dans', 'un', 'video', 'se', 'que', 'pour', 'elsevier','automatic','levels', 'expression', 'increased', 'effect', 'days', 'high','however','index', 'significantly', 'af', 'function'}
# Add the new words to the existing set
stop_words.update(new_stop_words)
# Building set of stop_words manually
#stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stop_words]
         for document in text_corpus]

#%%
"""Lemmatize, make bigram and trigram texts"""
# Lemmatize function using WordNet Lemmatizer
def lemmatize_text(text):
    return [get_lemma(word) for word in text]
# Tokenization
tokenized_texts = [word_tokenize(doc.lower()) for doc in text_corpus]
# Lemmatization
lemmatized_texts = [lemmatize_text(tokens) for tokens in tokenized_texts]

# Bigram and Trigram creation
bigram = Phrases(lemmatized_texts, min_count=5, threshold=100)
trigram = Phrases(bigram[lemmatized_texts], min_count=5, threshold=100)
bigram_phraser = Phraser(bigram)
trigram_phraser = Phraser(trigram)

# Apply bigram and trigram models to texts
bigram_texts = [bigram_phraser[text] for text in lemmatized_texts]
trigram_texts = [trigram_phraser[bigram_phraser[text]] for text in lemmatized_texts]
# Remove stopwords
cleaned_texts = [[word for word in text if word not in stop_words] for text in trigram_texts]

#%%
""" Tokenize and Build frame of words to being processed on LDA"""
# Define a function to filter tokens by POS tags
def filter_pos(tokens, allowed_pos_tags):
    tagged_tokens = nltk.pos_tag(tokens)
    return [token for token, tag in tagged_tokens if tag in allowed_pos_tags]

# Allowed POS tags based on your requirements
#allowed_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
# Proper nouns refer to specific names, organizations, or places. 
allowed_pos_tags = ['NNP', 'NNPS']


# Apply POS filtering
filtered_texts = [filter_pos(text, allowed_pos_tags) for text in cleaned_texts]

# Count word frequencies
frequency = defaultdict(int)
for text in filtered_texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in filtered_texts]
pprint.pprint(processed_corpus)

# Create a Dictionary for the processed corpus
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

# Convert the dictionary to a bag-of-words corpus for reference.
corpus = [dictionary.doc2bow(text) for text in processed_corpus]

"""
# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in cleaned_texts:
    for token in text:
        frequency[token] += 1 # Increase frecuency Token or decrease depending espected results
# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)
# Attempting to directly pass processed_corpus to LdaModel without first converting it to a Bag-of-Words (BoW) format using the doc2bow method for each document
# Create Dictionary for processed corpus
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)
# Convert the dictionary to a bag-of-words corpus for reference.
corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# Concept similares"""
#%%
""" Train an LDA model using a Gensim corpus """

number_topics = 8
number_words = 12

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
# Train the LDA model on the BoW corpus.
#lda = LdaModel(corpus, num_topics=10)
# Train the LDA model
lda_model = LdaModel(corpus, num_topics=number_topics, id2word=dictionary, passes=30)
#lda = models.LdaModel(corpus, num_topics=10)

#%%
""" Print LDA Topic Modeling vector """
# Iterate through each topic
for topic_number, topic in lda_model.print_topics(num_topics=number_topics, num_words=number_words):
    print("Topic Number:", topic_number)
    print("Original Topic String:", topic)
    # Extract words and their weights
    words = topic.split('+')
    print("Split Words:", words)
    # Remove the weights and keep only the words
    words = [word.split('*') for word in words if '*' in word]
    words = [word[1].replace('"', '').strip() for word in words]
    print("Processed Words:", words)
    # Create a readable string for each topic
    topic_str = ", ".join(words)
    print(f"Topic {topic_number}: {topic_str}")

# Create a visualization
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis_data)

#%%

# Print LDA Topic Modeling vector
topics = lda_model.print_topics(num_words=number_words)
for topic in topics:
    print(topic)

#%% 
""" Generate human-readable topic names """
# Iterate through each topic
for topic_number, topic in lda_model.print_topics(num_topics=number_topics, num_words=number_words):
    # Extract words and their weights
    words = topic.split('+')
    # Remove the weights and keep only the words
    # Remove the weights and keep only the words, while handling the IndexError
    words = [word.split('*')[1].replace('"', '').strip() for word in words if len(word.split('*')) > 1]
    # Create a readable string for each topic
    topic_str = ", ".join(words)
    print(f"Topic {topic_number}: {topic_str}")
# %%
