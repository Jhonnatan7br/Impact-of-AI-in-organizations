""" BERT Model: All requirements and optimization is defined on Pytorch and hugging face official documentation"""

Hugging_face_BERT_documentation = 'https://huggingface.co/docs/transformers/model_doc/bert'
Pytorch_transformers = 'https://pytorch.org/hub/huggingface_pytorch-transformers/'

#%%
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import numpy as np

""" Load pre-trained BERT model and tokenizer """

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
num_categories = 3  # Replace with the actual number of categories in your dataset
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_categories)

# Assuming there is a list of abstracts (texts) and their corresponding labels (categories)
research = pd.read_csv("C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/Datasets/scopus.csv")

# If 'Label' is not available, replace it with your actual label column name
sub_dataset = research.head(5251)
text_corpus = sub_dataset['Abstract'].tolist()

# Create an empty list to store the labels
labels = ['technology','medicine','energy']
# Convert labels list to a pandas Series and assign it to the 'Label' column in sub_dataset
sub_dataset['Label'] = pd.Series(labels)
#labels = sub_dataset['Label'].tolist()
#%%
""" Tokenizing preprocesed Data"""
# Initialize empty lists to store tokenized input and attention masks
input_ids = []
attention_masks = []

# Define a maximum sequence length
max_length = 128  # You can adjust this value based on your needs

# Tokenize and preprocess your dataset
for text in text_corpus:
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])
#%%

"""Encoding Labels"""
# Initialize the LabelEncoder

label_encoder = LabelEncoder()

# Fit the encoder on the labels and transform the labels into numerical values
encoded_labels = label_encoder.fit_transform(labels)

# Get the number of sequences from the length of the input_ids list
num_sequences = len(input_ids)

# Generate labels for your input data by repeating the encoded labels
expanded_labels = np.tile(encoded_labels, num_sequences // len(labels) + 1)[:num_sequences]

# Convert the numerical labels to a PyTorch tensor
labels_tensor = torch.tensor(expanded_labels)

# Ensure the shape of the labels matches the number of rows in your input data
assert labels_tensor.shape[0] == num_sequences
#%%
""" Procesing Encoding labels on the Tensor (It has to had the same size as attention_mask and input_ids)"""
# Assuming input_ids and attention_masks are lists of tensors
# Calculate the lengths of each sequence
sequence_lengths = [len(seq) for seq in input_ids]

# Sort the input sequences by length in descending order
sorted_indices = sorted(range(len(sequence_lengths)), key=lambda i: sequence_lengths[i], reverse=True)

# Filter out any indices that are out of range
sorted_indices = [i for i in sorted_indices if i < len(input_ids)]

# Reorder input_ids, attention_masks, and labels based on sorted_indices
labels = torch.tensor(encoded_labels)
input_ids = [input_ids[i] for i in sorted_indices]
attention_masks = [attention_masks[i] for i in sorted_indices]

# Create a mask for valid indices in sorted_indices
valid_indices_mask = [i < len(encoded_labels) for i in sorted_indices]

# Filter sorted_indices to contain only valid indices
sorted_indices = [i for i, valid in zip(sorted_indices, valid_indices_mask) if valid]

# Filter labels to contain only valid labels
labels = encoded_labels[sorted_indices]

# Debug: Print the lengths after reordering
print("Input IDs Length (After Reordering):", len(input_ids))
print("Attention Masks Length (After Reordering):", len(attention_masks))
print("Labels Length (After Reordering):", len(labels))

#%%
"""Convert lists of tensors to a single tensor"""
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)


tensor = torch.from_numpy(labels)
# Repeat the values in the labels tensor to match the size of the other tensors
tensor = tensor.repeat(input_ids.size()[0])
# Unsqueeze the tensor tensor to match the dimensions of input_ids
tensor1 = tensor.unsqueeze(1).unsqueeze(1)
# USE A COPY OF TENSOR
"""
resulting in a tensor of shape 
tensor.size()
torch.Size([15753, 1, 1])

It does not match with  input_ids of size
torch.Size([5251, 1, 128])
"""
#%%
""" Create data loaders or tensor dataset"""
dataset = TensorDataset(input_ids, attention_masks, tensor1)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

#%%

"""Training BERT Model"""
# Training loop
num_epochs = 1 # Augment epochs to gain efficiency on the model
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate on the validation or test set
# You can use a similar data loader setup as above and compute metrics like accuracy.

# %%
        
""" Save important variables, tensor information and dataset on the cache """
import os
import torch
import numpy as np
import pickle

# Define the cache directory
cache_dir = 'C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/1. NLP (Narural Language procesing)/cache '  # Change this to the actual path on your machine

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Define file paths
expanded_labels_path = os.path.join(cache_dir, 'expanded_labels.npy')
input_ids_path = os.path.join(cache_dir, 'input_ids.pt')
labels_tensor_path = os.path.join(cache_dir, 'labels_tensor.pt')
labels_path = os.path.join(cache_dir, 'labels.pkl')

# Save the expanded_labels as a NumPy file
np.save(expanded_labels_path, expanded_labels)

# Save the input_ids as a PyTorch file (assuming it's a list of tensors)
torch.save(input_ids, input_ids_path)

# Save the labels_tensor as a PyTorch file
torch.save(labels_tensor, labels_tensor_path)

# Save the labels using pickle
with open(labels_path, 'wb') as f:
    pickle.dump(labels, f)

#%%
"""Save the model"""

from transformers import BertForSequenceClassification

# Assuming `model` is an instance of BertForSequenceClassification and is already trained
# ...

# Define the model directory path
models_dir = 'C:/Users/Jhonnatan/Documents/GitHub/Impact-of-AI-in-organizations/1. NLP (Narural Language procesing)/models'

# Create the models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Save the trained model
model.save_pretrained(models_dir)

# The model's configuration and weights will be saved in the 'models' directory.
# You will find files like `config.json` and `pytorch_model.bin` there.
