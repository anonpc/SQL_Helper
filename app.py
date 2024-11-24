from flask import Flask, request, render_template
from models.t5_model import T5_SQL_Model

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Инициализация модели
model = T5_SQL_Model(model_path='anonpc/SQL_HelperT5', tokenizer_path='anonpc/SQL_HelperT5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    description = request.form.get('description')
    context = request.form.get('context')

    if not description or not context:
        return render_template(
            'generate_result.html',
            description="Описание отсутствует",
            generated_query="Ошибка: контекст не задан",
        )

    try:
        generated_query = model.generate(description, context)
    except Exception as e:
        generated_query = f"Ошибка при генерации: {e}"

    return render_template(
        'generate_result.html',
        description=description,
        context=context,
        generated_query=generated_query,
    )

if __name__ == '__main__':
    app.run(debug=True)