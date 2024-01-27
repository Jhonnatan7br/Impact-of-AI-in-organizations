import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

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

# Now, all sequences will have the same length of 'max_length'
# You can concatenate the tensors without any issues
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Create data loaders
dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
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
