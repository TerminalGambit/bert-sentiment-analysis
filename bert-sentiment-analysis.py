from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model, with a classification layer on top for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')