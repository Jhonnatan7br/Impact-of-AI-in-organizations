#%%
# Import requirements
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import nltk
import spacy
import os

#%%
""" Load spaCy model for preprocessing"""
nlp = spacy.load('en_core_web_sm')

# Load your dataset
research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")

# Create a sub-dataset with the first 5251 rows
sub_dataset = research.head() # Config Num of data instances

#%%
""" Extract abstracts from the 'Abstract' column of the dataframe """
text_corpus = sub_dataset['Abstract'].tolist()

# Preprocess the text data using spaCy and remove stopwords
stop_words = set(stopwords.words('english'))
new_stop_words = {'(ai)', 'patien', 'patients', '=', 'de', 'control', 'model', 'system', 'et', 'results', 'power', '±', 'new', 'compared', 'risk', 'data', 'smart', 'la', 'abstract', 'om', '“no', 'available', 'set', 'problem', 'teatures', 'siven', 'class', 'rights', 'general', 'cows', 'milk', 'relevant', 'reserved."', 'time', 'pattern', 'constraint', 'classiters', 'vve I', 'problems', 'consider', 'propositional', 'logic', 'present', 'space', 'large', 'springer', 'prove', 'intake', 'la', '©', 'found', 'problem', 'features', 'given', 'p', '(p)', 'knowledge', 'results', 'si', 'd', '"[no', 'available]"', 'use', 'field', '"The', 'provide', 'based', 'paper', 'propose', 'decision', '2021', 'process', 'methods', 'paper,', 'however', 'number', 'studies', 'study', '<', 'conclusion:', '2', '1', 'significant', 'included', 'total', '(n', 'des', 'les', 'genetic', 'à', 'en', 'le', 'une', 'qui', 'du', 'associated', 'literature', 'review', ',', 'intelligence', 'artificial', 'approach', 'proposed', 'intelligence', 'accuracy', 'parameters', 'group', 'methods:', 'results:', '(p', 'low', 'different', 'higher', 'analysis', 'published', 'articles', 'coding', 'dans', 'un', 'video', 'se', 'que', 'productivity', 'pour', 'elsevier', 'automatic', 'levels', 'expression', 'increased', 'effect', 'days', 'high', 'however', 'index', 'significantly', 'af', 'function'}
stop_words.update(new_stop_words)

preprocessed_corpus = []

for doc in text_corpus:
    # Tokenize and preprocess using spaCy
    doc_tokens = [token.lemma_.lower() for token in nlp(doc) if token.lemma_.lower() not in stop_words]
    preprocessed_corpus.append(doc_tokens)


# %%
""" Train Word2Vec model on the preprocessed corpus"""
model = Word2Vec(sentences=preprocessed_corpus, vector_size=100, window=5, min_count=1, sg=0)

# Define the file path to save the model
model_path = "1. NLP (Natural Language Processing)/word2vec_model"

# Ensure the directory exists, if not, create it
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the Word2Vec model
model.save(model_path)

# Load the Word2Vec model
model = Word2Vec.load(model_path)

# Get word vectors
vector = model.wv['ai']  # Replace 'word1' with the word you want to get the vector for
print(vector)

# Create a DataFrame with a single column named 'Vector'
df = pd.DataFrame({'Vector': [vector]})

# Display the DataFrame
print(df)
#%%
""" Requirements to creade nodes & Get word vectors and words"""
import networkx as nx
import matplotlib.pyplot as plt

#%%
""" Create a graph"""

# Create a graph
G = nx.Graph()

# Add nodes with word vectors as attributes
for word in model.wv.key_to_index:
    G.add_node(word, vec=model.wv[word])
#%%
""" Add edges based on similarity between word vectors"""
for i, word1 in enumerate(model.wv.index_to_key):
    for j, word2 in enumerate(model.wv.index_to_key):
        if i < j:
            similarity = model.wv.similarity(word1, word2)
            if similarity > 0.7:  # Adjust threshold as needed
                G.add_edge(word1, word2, weight=similarity)

# Visualize the graph
pos = nx.spring_layout(G)  # Compute layout
plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_size=20)
plt.show()

# %%
# Placeholder for category_words
category_words = {
    'category1': ['ai', 'supply', 'logistic'],
    'category2': ['ai', 'marketing', 'selling'],
    'category3': ['ai', 'finance', 'bank']
}  # Add actual words for each category

# Assign colors to nodes based on categories
category_colors = {
    'category1': 'red',
    'category2': 'blue',
    'category3': 'green'
}  # Add more categories and colors as needed

# Assign node sizes based on node degree
node_sizes = {node: degree * 100 for node, degree in dict(G.degree()).items()}

# Visualize the graph
plt.figure(figsize=(12, 10))

# Draw nodes
for category, color in category_colors.items():
    nodes = [node for node in G.nodes() if node in category_words.get(category, [])]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=[node_sizes[node] for node in nodes], label=category)

# Draw edges
nx.draw_networkx_edges(G, pos, width=[data['weight'] * 5 for _, _, data in G.edges(data=True)], alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10)

# Add legend
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for label, color in category_colors.items()]
plt.legend(handles=legend_handles, title='Categories', loc='upper right')

# Show plot
plt.title('Word2Vec Graph with Categories and Labels')
plt.axis('off')
plt.show()

# %%
