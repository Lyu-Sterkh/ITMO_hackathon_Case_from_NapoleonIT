import cv2
import re
import telebot
import numpy as np
import pickle
import pytesseract
from telebot import types
from scipy.ndimage import rotate
import time
import requests
import zipfile
import io
import csv
from datetime import datetime

# Введите свой API_TOKEN
API_TOKEN = 'API_TOKEN'
bot = telebot.TeleBot(API_TOKEN)
bot.timeout = 300

# Функция send_request_with_retry принимает на вход сессию, URL, метод (по умолчанию 'GET') и другие аргументы.
def send_request_with_retry(sess, url, method='GET', **kwargs):
    # Задаем число попыток на выполнение запроса.
    retries = 5
    # Задаем начальную задержку перед повтором запроса в секундах.
    delay = 2
    # Цикл for выполняется заданное число попыток (retries).
    for i in range(retries):
        try:
            # Выполнение запроса с заданным методом (method).
            response = sess.request(
                method,
                url,
                # Устанавливаем заголовки по умолчанию для библиотеки telebot.
                headers=telebot.apihelper.default_headers(),
                # Устанавливаем таймауты для подключения и чтения из библиотеки telebot.
                timeout=(telebot.apihelper.connect_timeout, telebot.apihelper.read_timeout),
                # Передаем дополнительные аргументы в запрос через троичный оператор (как опциональные параметры).
                **kwargs,
            )
            # Проверяем статус ответа. В случае ошибки поднимаем исключение.
            response.raise_for_status()
            # Если ответ корректен, возвращаем его.
            return response
        except requests.RequestException:
            # Если возникло исключение, проверяем, не последняя ли это попытка.
            if i < (retries - 1):
                # Если попытка не последняя, ждем заданное время перед повтором запроса.
                time.sleep(delay)
                # Увеличиваем задержку перед следующим повтором запроса в два раза.
                delay *= 2
            else:
                # Если все попытки исчерпаны, поднимаем исключение.
                raise

telebot.apihelper.execute = send_request_with_retry

photos_directory = 'incoming_photo/'

# Устанавливаем путь к исполняемому файлу Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Загрузите предварительно сохраненные модели для предсказания аналогов
with open('models/model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Функция изменения и очистки текста
def preprocess(text):
    text = text.lower()
    text = text.replace('ё', 'е')
    text = text.replace('ъ', 'ь')
    text = text.replace('й', 'и')
    text = re.sub('[^а-яА-яa-zA-Z0-9 ]', ' ', text)
    return text

# Определяем функцию, которая определит лучший угол поворота изображения для улучшения результатов OCR
def determine_best_rotation_angle(image, lang='rus+eng', rotation_range=(-10, 10), steps=1):
    current_max_confidence = -np.inf
    best_rotation_angle = 0
    for rotation_angle in range(rotation_range[0], rotation_range[1] + steps, steps):
        rotated_image = rotate(image, rotation_angle, reshape=False)
        configs_data = pytesseract.image_to_data(rotated_image, output_type=pytesseract.Output.DICT, lang=lang, config='--psm 6')
        mean_confidence = np.mean(np.asarray(configs_data["conf"], dtype=float))
        if mean_confidence > current_max_confidence:
            best_rotation_angle = rotation_angle
            current_max_confidence = mean_confidence
    return best_rotation_angle

# Предобработка и извлечение текста с помощью tesseract
def get_text_from_photo(photo):

    # Читаем и приводим к единому размер изображения
    image = cv2.imdecode(np.asarray(bytearray(photo), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    target_width = 1600
    target_height = 1600
    scale_factor = min(target_width / image.shape[1], target_height / image.shape[0])
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Конвертируем цветное изображение в оттенки серого
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Применяем размытие по Гауссу для уменьшения шума
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Нормализуем уровень яркости изображения
    normalized_image = cv2.normalize(blurred_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Определяем лучший угол поворота
    best_rotation_angle = determine_best_rotation_angle(normalized_image)

    # Поворачиваем изображение на лучший угол поворота
    rotated_image = rotate(normalized_image, best_rotation_angle, reshape=False)

    # Обрезаем верхнюю часть изображения (35%)
    cropped_image = rotated_image[:int(rotated_image.shape[0] * 0.35), :]

    # Применяем пороговое преобразование Оцу для бинаризации изображения
    _, binary_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Применяем морфологическое преобразование (Дилатация, Замыкание)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
    morphed_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Указываем параметры OCR
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(morphed_image, config=custom_config, lang='rus+eng')

    return text

# Обработка текста и классификация товара
def classify_product(text):
    text = preprocess(text)
    comparison_data = [text]
    comparison_data_vectorized = vectorizer.transform(comparison_data)
    predicted_labels = classifier.predict(comparison_data_vectorized)
    return predicted_labels[0]

# Функционал для обработки архива и сохранения результатов в CSV
def process_zip_file(zip_data, output_file):
    csv_data = [["File Name", "Prediction"]]
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zipped_file:
        for file_name in zipped_file.namelist():
            if file_name.lower().endswith('.jpg'):
                with zipped_file.open(file_name, 'r') as img_file:
                    img_data = img_file.read()
                    text = get_text_from_photo(img_data)
                    prediction = classify_product(text)
                    if prediction == 1:
                        prediction_str = "Предсказано как аналог"
                    else:
                        prediction_str = "Не предсказано как аналог"
                    csv_data.append([file_name, prediction_str])

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerows(csv_data)

# Добавление новой кнопки, кнопка "Загрузить фотографии в zip файле"
@bot.message_handler(commands=['start'])
def start_handler(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton('Активировать фотокамеру'))
    markup.add(types.KeyboardButton('Загрузить фотографии в zip файле'))  # добавляем новую кнопку
    bot.send_message(message.chat.id, 'Выбери команду:', reply_markup=markup)


# Функция обработки сообщения с архивом
@bot.message_handler(content_types=['document'])
def document_handler(message):
    if message.document.mime_type == 'application/zip':
        file_info = bot.get_file(message.document.file_id)
        zip_data = bot.download_file(file_info.file_path)
        output_csv_path = 'csv_files_from_zip/result_{}_{}.csv'.format(message.chat.id, datetime.now().strftime("%Y%m%d%H%M%S"))
        process_zip_file(zip_data, output_csv_path)

        with open(output_csv_path, 'rb') as result_csv:
            bot.send_document(message.chat.id, result_csv)
    else:
        bot.reply_to(message, 'Пожалуйста, отправьте файл архива в формате ZIP.')


@bot.message_handler(content_types=['photo'])
def photo_handler(message):
    photo = message.photo[-1].file_id
    file_info = bot.get_file(photo)
    downloaded_file = bot.download_file(file_info.file_path)

    # Получаем текст с фотографии и классифицируем его
    text = get_text_from_photo(downloaded_file)
    result = classify_product(text)

    # Возвращаем результат классификации пользователю
    if result == 1:
        answer = "Предсказано как аналог"
    else:
        answer = "Не предсказано как аналог"
    bot.reply_to(message, answer)

bot.polling()