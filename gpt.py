import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Очистка памяти CUDA (если нужно)
torch.cuda.empty_cache()
import re
# Загружаем модель и токенизатор
model_name_or_path = "models/fine_tuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to("cpu")  # Переключаем на CPU

# Функция для генерации текста
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, num_beams=3):
    # Токенизируем входной текст
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Генерируем текст с учетом параметров
    with torch.no_grad():  # Отключаем градиенты для экономии памяти
        output = model.generate(
            input_ids=inputs["input_ids"],  # Передаем только input_ids
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Избегаем повторов
            pad_token_id=tokenizer.eos_token_id,
            num_beams=num_beams,  # Добавляем beam search
            early_stopping=True  # Останавливаемся, если все лучи нашли токен конца
        )

    # Декодируем результат в текст
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

import re

def truncate_after_last_uppercase(text):
    # Ищем последнюю заглавную букву и обрезаем текст до нее
    match = re.search(r'[A-ZА-Я](?=[^A-ZА-Я]*$)', text)
    if match:
        # Обрезаем текст до найденной буквы, не включая ее
        return text[:match.start()]
    return text  # Если заглавных букв нет, возвращаем исходный текст



def add_periods_and_capitalize(text):
    # Добавляем точку в конце предложения, если её нет
    text = re.sub(r'([а-яa-zа-яё]{1})(\s*)([A-ZА-ЯЁ])', r'\1.\2\3', text)  # Для предложений в середине текста

    # Если текст не заканчивается точкой, добавляем её
    if not text.endswith('.'):
        text += '.'

    # Приводим каждое новое предложение к заглавной букве
    text = re.sub(r'([.!?])(\s+)([a-zа-яё])', lambda match: match.group(1) + match.group(2) + match.group(3).upper(), text)

    return text




# Заголовок и описание приложения
st.title("Генерация текста с GPT-3 (дообученная модель)")
st.markdown("Введите текст, чтобы модель продолжила его. Используйте различные параметры для настройки генерации.")

# Пользовательский ввод (промпт)
prompt = st.text_area("Введите текст для продолжения:", "Как будет развиваться будущее?")

# Параметры генерации текста
max_length = st.slider("Максимальная длина текста", min_value=10, max_value=200, value=50)
temperature = st.slider("Температура", min_value=0.0, max_value=1.0, value=0.7)
top_k = st.slider("Top-k", min_value=0, max_value=100, value=50)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.95)
beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=3)

# Кнопка для запуска генерации
if st.button("Сгенерировать текст"):
    with st.spinner("Генерация..."):
        try:
            generated_text = generate_text(prompt, max_length, temperature, top_k, top_p, num_beams=beam_size)
            clean_text = truncate_after_last_uppercase(generated_text)
            text_with_periods = add_periods_and_capitalize(clean_text)
            st.subheader("Сгенерированный текст:")
            st.write(text_with_periods)
        except Exception as e:
            st.error(f"Ошибка генерации текста: {e}")


