import sys
import os
import transformers
from read_conll import read_conll
from prepare_dataset import prepare_dataset
from compute_matrics import compute_matrics
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python executable:", sys.executable)
print("Transformers imported from:", transformers.__file__)
print("Transformers version:", transformers.__version__)

print("TrainingArguments location:", TrainingArguments.__module__)
print("TrainingArguments class file:", getattr(TrainingArguments, '__file__', 'N/A'))


#Load data
address = "C:/Users/hp/Desktop/Kifya/week_4/Data/Cleaned/"
sentences, labels = read_conll(address + "cleaned_telegram_data.csv")

#Tokenize and prepare dataset
tokenized_dataset, tokenizer, label2id, id2label = prepare_dataset(sentences, labels)

#Load model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

#Define training arguments
training_args = TrainingArguments(
    output_dir="C:/Users/hp/Desktop/Kifya/week_4/model",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save model at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="C:/Users/hp/Desktop/Kifya/week_4/log",
)

print("TrainingArguments created successfully.")

#Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

print("Trainer created successfully.")

#Train the model
trainer.train()

#Save the model and tokenizer
trainer.save_model("C:/Users/hp/Desktop/Kifya/week_4/model")
tokenizer.save_pretrained("C:/Users/hp/Desktop/Kifya/week_4/tokenizer")