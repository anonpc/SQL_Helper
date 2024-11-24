# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# from torch.utils.data import DataLoader
# from dataset.SQLDataset import SQLDataset

# def collate_fn(batch, tokenizer):
#     """
#     Функция коллаборации данных для DataLoader.

#     Args:
#         batch (list): Пакет данных из датасета.
#         tokenizer: Токенизатор модели.

#     Returns:
#         dict: Батч токенизированных данных.
#     """

#     inputs = [item["input"] for item in batch]
#     outputs = [item["output"] for item in batch]

#     inputs_enc = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
#     outputs_enc = tokenizer(outputs, truncation=True, padding=True, return_tensors="pt")

#     labels = outputs_enc.input_ids
#     labels[labels == tokenizer.pad_token_id] = -100  # Игнорируем токены PAD для потерь

#     return {
#         "input_ids": inputs_enc.input_ids,
#         "attention_mask": inputs_enc.attention_mask,
#         "labels": labels
#     }

# def train():
#     # Пути к данным и параметрам
#     train_data_path = "/Users/dmitrijvarlygin/Intelligent_assistant_for_administering_SQL_queries/generation_data.json"
#     val_data_path = "/Users/dmitrijvarlygin/Intelligent_assistant_for_administering_SQL_queries/generation_data.json"

#     # Инициализация модели и токенизатора
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")

#     # Создаем датасеты
#     train_dataset = SQLDataset(train_data_path)
#     val_dataset = SQLDataset(val_data_path)

#     # Гиперпараметры обучения
#     training_args = TrainingArguments(
#         output_dir="t5_sql_fine_tuned",
#         evaluation_strategy="steps",
#         eval_steps=500,
#         save_steps=1000,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         learning_rate=3e-5,
#         num_train_epochs=3,
#         logging_dir="logs",
#         logging_steps=50,  # Более частое логирование
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         report_to="none"  # Отключение интеграции с WandB/MLFlow
#     )


#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         data_collator=lambda batch: collate_fn(batch, tokenizer),
#         compute_metrics=None  # Если не нужна метрика
#     )


#     trainer.train()
#     model.save_pretrained("t5_sql_model_finetuned")
#     tokenizer.save_pretrained("t5_sql_model_finetuned")

# if __name__ == "__main__":
#     train()

# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
# from torch.utils.data import DataLoader
# from dataset.SQLDataset import SQLDataset

# def collate_fn(batch, tokenizer):
#     """
#     Функция коллаборации данных для DataLoader.

#     Args:
#         batch (list): Пакет данных из датасета.
#         tokenizer: Токенизатор модели.

#     Returns:
#         dict: Батч токенизированных данных.
#     """
#     inputs = [item["input"] for item in batch]
#     outputs = [item["output"] for item in batch]

#     max_input_length = 128
#     max_output_length = 64
#     inputs_enc = tokenizer(inputs, truncation=True, padding=True, max_length=max_input_length, return_tensors="pt")
#     outputs_enc = tokenizer(outputs, truncation=True, padding=True, max_length=max_output_length, return_tensors="pt")


#     # inputs_enc = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
#     # outputs_enc = tokenizer(outputs, truncation=True, padding=True, return_tensors="pt")

#     labels = outputs_enc.input_ids
#     labels[labels == tokenizer.pad_token_id] = -100  # Игнорируем токены PAD для потерь

#     return {
#         "input_ids": inputs_enc.input_ids,
#         "attention_mask": inputs_enc.attention_mask,
#         "labels": labels
#     }

# def train():
#     # Пути к данным
#     train_data_path = "/Users/dmitrijvarlygin/Intelligent_assistant_for_administering_SQL_queries/generation_data.json"
#     val_data_path = "/Users/dmitrijvarlygin/Intelligent_assistant_for_administering_SQL_queries/generation_data.json"

#     # Инициализация модели и токенизатора
#     tokenizer = T5Tokenizer.from_pretrained("t5-small")
#     model = T5ForConditionalGeneration.from_pretrained("t5-small")

#     # Создание датасетов и DataLoader
#     train_dataset = SQLDataset(train_data_path)
#     val_dataset = SQLDataset(val_data_path)

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True,
#         collate_fn=lambda batch: collate_fn(batch, tokenizer)
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=8,
#         shuffle=False,
#         collate_fn=lambda batch: collate_fn(batch, tokenizer)
#     )

#     # Оптимизатор
#     optimizer = AdamW(model.parameters(), lr=1e-5)

#     # Устройство (GPU/CPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Цикл обучения
#     num_epochs = 3
#     model.train()

#     for epoch in range(num_epochs):
#         print(f"Эпоха {epoch + 1}/{num_epochs}")
#         for batch_idx, batch in enumerate(train_dataloader):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             # Прямой проход
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
#             loss = outputs.loss

#             # Обратный проход
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch_idx % 50 == 0:
#                 print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

#         # Оценка на валидационной выборке
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_dataloader:
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["labels"].to(device)

#                 outputs = model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )
#                 val_loss += outputs.loss.item()

#         val_loss /= len(val_dataloader)
#         print(f"Validation Loss: {val_loss:.4f}")
#         model.train()

#     # Сохранение модели и токенизатора
#     model.save_pretrained("t5_sql_model_finetuned")
#     tokenizer.save_pretrained("t5_sql_model_finetuned")

# if __name__ == "__main__":
#     train()

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
