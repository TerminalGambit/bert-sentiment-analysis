from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datasets import load_dataset

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model, with a classification layer on top for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load dataset
dataset = load_dataset('imdb')

# Preprocess data: tokenize and align labels with BERT inputs
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
