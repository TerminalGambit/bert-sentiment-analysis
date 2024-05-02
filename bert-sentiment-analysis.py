from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer

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

# Set format for pytorch tensors
tokenized_datasets.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
