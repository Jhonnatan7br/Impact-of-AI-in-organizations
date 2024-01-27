import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_categories)

# Tokenize and preprocess your dataset
# Assuming you have a list of abstracts (texts) and their corresponding labels (categories)
texts = ["Your abstracts go here"]
labels = [0, 1, 2]  # Replace with your actual labels

# Tokenize and convert to tensors
input_ids = []
attention_masks = []
for text in texts:
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

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
