from sklearn.model_selection import train_test_split
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Subset
from dataset.SQLDataset import SQLDataset

def collate_fn(batch, tokenizer, max_input_length=128, max_output_length=64):
    inputs = [item["input"] for item in batch]
    outputs = [item["output"] for item in batch]

    inputs_enc = tokenizer(inputs, truncation=True, padding=True, max_length=max_input_length, return_tensors="pt")
    outputs_enc = tokenizer(outputs, truncation=True, padding=True, max_length=max_output_length, return_tensors="pt")

    labels = outputs_enc.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": inputs_enc.input_ids,
        "attention_mask": inputs_enc.attention_mask,
        "labels": labels
    }

def train():
    data_path = "/Users/dmitrijvarlygin/Intelligent_assistant_for_administering_SQL_queries/generation_data.json"

    # Загрузка датасета
    dataset = SQLDataset(data_path)

    # Разделение на train/val
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # DataLoader
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataloader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # Модель и оптимизатор
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Обучение
    num_epochs = 3
    model.train()
    for epoch in range(num_epochs):
        print(f"Эпоха {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")
        model.train()

    # Сохранение модели
    model.save_pretrained("t5_sql_model_finetuned")
    tokenizer.save_pretrained("t5_sql_model_finetuned")

if __name__ == "__main__":
    train()
