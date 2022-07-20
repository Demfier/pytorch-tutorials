import os, numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, DatasetDict


def read_txt(file_path):
    with open(file_path, "r") as f:
        return [l.strip() for l in f.readlines()]


def get_dataset(data_dir):
    pos_lines = read_txt(os.path.join(data_dir, "pos.txt"))[:5000]
    neg_lines = read_txt(os.path.join(data_dir, "neg.txt"))[:5000]
    lines = pos_lines + neg_lines
    labels = [1] * len(pos_lines) + [0] * len(neg_lines)
    dataset = list(zip(lines, labels))
    np.random.shuffle(dataset)
    lines, labels = zip(*dataset)

    n_train = int(0.8 * len(lines))
    n_val = int(0.1 * len(lines))

    return DatasetDict(
        train=Dataset.from_dict({"text": lines[:n_train], "label": labels[:n_train]}),
        val=Dataset.from_dict(
            {
                "text": lines[n_train : n_train + n_val],
                "label": labels[n_train : n_train + n_val],
            }
        ),
        test=Dataset.from_dict({"text": lines[-n_val:], "label": labels[-n_val:]}),
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    data_dict = get_dataset("data")
    print(data_dict)

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)

    def preprocess(example):
        return tokenizer(example["text"], max_length=50, truncation=True)

    encoded_dataset = data_dict.map(preprocess, batched=True)

    backbone = AutoModelForSequenceClassification.from_pretrained(
        "bert-large-uncased", num_labels=2
    )
    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        "checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        num_train_epochs=5,
        metric_for_best_model="accuracy",
        per_device_eval_batch_size=64,
        per_device_train_batch_size=64,
    )

    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer
    trainer = Trainer(
        backbone,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(
        trainer.evaluate(eval_dataset=encoded_dataset["test"], metric_key_prefix="test")
    )


if __name__ == "__main__":
    main()
