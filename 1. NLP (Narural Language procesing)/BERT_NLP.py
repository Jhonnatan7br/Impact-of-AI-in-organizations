#%%
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import numpy as np

# Load pre-trained BERT model and tokenizer
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

# Now, all sequences will have the same length of 'max_length'
# You can concatenate the tensors without any issues
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
# Repeat each label for the corresponding number of sequences

# Initialize the LabelEncoder

label_encoder = LabelEncoder()

# Fit the encoder on the labels and transform the labels into numerical values
encoded_labels = label_encoder.fit_transform(labels)

# Generate labels for your input data by repeating the encoded labels
num_sequences = input_ids.size(0)
expanded_labels = np.tile(encoded_labels, num_sequences // len(labels) + 1)[:num_sequences]

# Convert the numerical labels to a PyTorch tensor
labels_tensor = torch.tensor(expanded_labels)

# Ensure the shape of the labels matches the number of rows in your input data
assert labels_tensor.shape[0] == num_sequences
#%%

labels = torch.tensor(labels)

# Find the maximum sequence length in input_ids and attention_masks
max_length = max(input_ids.size(0), attention_masks.size(0))

# Pad the shorter tensor to match the maximum length
if input_ids.size(0) < max_length:
    input_ids = F.pad(input_ids, (0, 0, 0, max_length - input_ids.size(0)))
elif attention_masks.size(0) < max_length:
    attention_masks = F.pad(attention_masks, (0, 0, 0, max_length - attention_masks.size(0)))

# Create data loaders or tensor dataset
dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

#%%
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
