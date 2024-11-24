import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5_SQL_Model:
    def __init__(self, model_path="anonpc/SQL_HelperT5", tokenizer_path="anonpc/SQL_HelperT5"):
        """
        Инициализация модели и токенизатора.
        
        Args:
            model_path (str): Путь к предобученной или дообученной модели T5.
            tokenizer_path (str): Путь к токенизатору.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.generate_model = T5ForConditionalGeneration.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generate_model.to(self.device)

    def generate(self, question, context):
        """
        Генерирует SQL-запрос на основе вопроса и контекста.

        Args:
            question (str): Текст вопроса.
            context (str): Описание таблицы (DDL).

        Returns:
            str: Сгенерированный SQL-запрос.
        """

        self.generate_model.eval()
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.generate_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        generated_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_query
    
        
    def fix(self, sql_query):
        """
        Исправляет SQL-запрос с ошибками.

        Args:
            sql_query (str): SQL-запрос с ошибками.

        Returns:
            str: Исправленный SQL-запрос.
        """
        self.generate_model.eval()
        input_text = f"fix this SQL-query: {sql_query}"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            ouputs = self.generate_model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        fixed_query = self.tokenizer.decode(ouputs[0], skip_special_tokens=True)
        return fixed_query

if __name__ == "__main__":
    # Пример использования модели
    sql_model = T5_SQL_Model(model_path="anonpc/SQL_HelperT5", tokenizer_path="anonpc/SQL_HelperT5")

    question = "How many heads of the departments are older than 56?"
    context = "CREATE TABLE head (age INTEGER)"

    print("Генерация SQL-запроса:")
    print(sql_model.generate(question, context))

    # incorrect_query = 'SEL name, age FROM users WHERE age > 18 ORDER BY name;'
    # print("Генерация правильного SQL-запроса:")
    # print(sql_model.fix(incorrect_query))
