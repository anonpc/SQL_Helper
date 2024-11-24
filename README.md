# SQL Helper

**SQL Helper** — это веб-приложение на Flask, которое позволяет генерировать SQL-запросы на основе описания задачи и контекста таблиц (DDL). Приложение использует предобученную модель T5 для генерации запросов.

## Возможности

- Генерация SQL-запросов на основе текстового описания.
- Удобный пользовательский интерфейс для ввода данных.
- Поддержка Flask для запуска веб-приложения.

## Используемые технологии
- **Python**
- **Flask**
- **PyTorch**
- **Transformers (Hugging Face)**
- **HTML & CSS**

## Используемая модель
Модель **T5** от Hugging Face была дообучена для задачи генерации SQL-запросов. 

### Процесс дообучения:
1. **Входные данные**:
   - Описание задачи: текстовый запрос пользователя.
   - Контекст: DDL для описания структуры таблиц.

2. **Выходные данные**:
   - Сгенерированный SQL-запрос.

3. **Детали обучения**:
   - Использована предобученная модель `t5-small`.
   - Гиперпараметры:
     - `learning_rate`: 1e-5
     - `batch_size`: 8
     - `num_epochs`: 3

4. Качество модели на валидационной выборке: 0.1466.
   
6. Дообученная модель доступна в репозитории Hugging Face по адресу: [anonpc/SQL_HelperT5](https://huggingface.co/anonpc/SQL_HelperT5)

---

## Установка и запуск

### 1. Клонирование репозитория
1. git clone [https://github.com/anonpc/SQL-Helper.git](https://github.com/anonpc/SQL_Helper.git)
2. cd SQL-Helper

### 2. Установка зависимостей
Создайте и активируйте виртуальное окружение, затем установите зависимости:

1. python -m venv venv
2. source venv/bin/activate  # Для Linux/macOS
3. venv\Scripts\activate     # Для Windows
4. pip install -r requirements.txt

### 3. Запуск приложения
Запустите Flask-приложение:
1. python app.py
2. Откройте браузер и перейдите по адресу http://127.0.0.1:5000/

---

### Как использовать?
1. Перейдите на главную страницу приложения.
2. Введите описание задачи в поле "Описание".
3. Введите DDL (определение таблиц) в поле "Контекст".
4. Нажмите "Сгенерировать SQL-запрос".
5. Получите результат — сгенерированный SQL-запрос.
