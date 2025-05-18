import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
from app.config import settings

def preprocess(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=settings.max_seq_length,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def main():
    data = load_dataset("rotten_tomatoes")

    model_checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    tokenized = data.map(lambda b: preprocess(b, tokenizer), batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=settings.model_dir,
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        eval_strategy="epoch",
        use_mps_device=(settings.device.type == "mps"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(settings.model_dir)
    tokenizer.save_pretrained(settings.model_dir)

if __name__ == "__main__":
    main()