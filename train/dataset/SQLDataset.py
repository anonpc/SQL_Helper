import json
from torch.utils.data import Dataset

class SQLDataset(Dataset):
    def __init__(self, file_path):
        """
        Инициализация датасета.

        Args:
            file_path (str): Путь к JSON-файлу с данными.
        """
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = f"question: {item['question']} context: {item['context']}"
        answer = item['answer']
        return {"input": question, "output": answer}