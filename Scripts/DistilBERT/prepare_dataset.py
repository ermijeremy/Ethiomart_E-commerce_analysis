from datasets import Dataset
from transformers import AutoTokenizer

def prepare_dataset(sentences, labels):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    unique_labels = sorted(set(tag for seq in labels for tag in seq))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        aligned_labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith("I-") else -100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": labels})
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    tokenized = dataset.map(tokenize_and_align_labels, batched=True)

    return tokenized, tokenizer, label2id, id2label
