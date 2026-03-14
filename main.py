import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from scipy import ndimage
import cv2 as cv2
import logging
import numpy as np
from tensorflow.keras.models import load_model
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
os.chdir(os.path.dirname(__file__)+"\\bot")
BOT_TOKEN = "8488000296:AAHRpbq-4yzl25F11TFwHeycIpQrRe7NoeQ"
bot = telebot.TeleBot(BOT_TOKEN)
bot.delete_my_commands()
bot.set_my_commands(['start'])
user_choices = {}
rest=load_model('bestaiever50epochmse.h5')

def ai(ph):
    d=np.array(cv2.imread(ph,cv2.IMREAD_GRAYSCALE)).astype("float32")
    d=np.where(d > 235.0, 255.0, d)
    cv2.imwrite('gt.jpg', d)
    d=d/255.0
    x, y = d.shape
    nx = (x + 127) // 128
    ny = (y + 127) // 128
    
    on = np.ones((nx * 128, ny * 128), dtype=d.dtype)
    on[0:x, 0:y] = d
    
    for i in range(nx):
        for j in range(ny):
            block = on[i*128:(i+1)*128, j*128:(j+1)*128]
            print(f"Блок {i},{j}: {block.shape}")
            block_expanded = block.reshape(1, 128, 128, 1)
            predicted = rest.predict(block_expanded)
            processed_block = predicted[0, :, :, 0] * 255.0
            on[i*128:(i+1)*128, j*128:(j+1)*128] = processed_block
    result = on[0:x, 0:y].astype(np.uint8)
    cv2.imwrite('snd.jpg', result)

def cv2r(ph):
    omask=np.array(cv2.imread(ph,cv2.IMREAD_GRAYSCALE))
    mask = np.where(omask > 240, 255, 0).astype(np.uint8)
    mask_255 = (mask == 255)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size))
    influence = ndimage.convolve(mask_255.astype(float), kernel, mode='constant')
    result = mask.copy()
    result[(mask == 0) & (influence > 0)] = 255
    pr = cv2.inpaint(cv2.imread(ph),result,3,cv2.INPAINT_TELEA)
    cv2.imwrite('snd.jpg',pr)

def create_main_menu():
    """Создает клавиатуру главного меню"""
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    button_color = KeyboardButton("CV2")
    button_bw = KeyboardButton("Нейросеть")
    keyboard.add(button_color, button_bw)
    return keyboard

@bot.message_handler(commands=['start'])
def start_command(message):
    """Обработчик команды /start"""
    user_id = message.from_user.id
    user_choices[user_id] = "color"
    
    bot.send_message(
        message.chat.id,
        "📷 **Бот для восстановления фотографий**\n\n"
        "Выберите режим обработки фото:\n"
        "• **CV2** - хорошо подходит для цветных фото с большим разрешением\n"
        "• **Нейросеть** - хорошо подходит для чёрно-белых фото с маленьким разрешением разрешением\n\n"
        "Просто отправьте фото!",
        reply_markup=create_main_menu(),
        parse_mode="Markdown"
    )


@bot.message_handler(func=lambda message: message.text in ["CV2", "Нейросеть"])
def handle_menu_buttons(message):
    """Обрабатывает нажатия кнопок меню"""
    user_id = message.from_user.id
    text = message.text
    
    if text == "CV2":
        user_choices[user_id] = "color"
        bot.send_message(
            message.chat.id,
            "✅ Установлен режим: **CV2**\n"
            "Теперь отправьте мне фото!",
            reply_markup=create_main_menu(),
            parse_mode="Markdown"
        )
    
    elif text == "Нейросеть":
        user_choices[user_id] = "bw"
        bot.send_message(
            message.chat.id,
            "✅ Установлен режим: **Нейросеть**\n"
            "Теперь отправьте мне фото!",
            reply_markup=create_main_menu(),
            parse_mode="Markdown"
        )
    else:
        bot.send_message(
            message.chat.id,
            "📋 **Текущий режим:** Нейросеть\n\n"
            "Выберите новый режим:",
            reply_markup=create_main_menu(),
            parse_mode="Markdown"
            )

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Обрабатывает входящие фото"""
    try:
        user_id = message.from_user.id
        color_mode = user_choices.get(user_id, "color")
        chat_id = message.chat.id
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with open("received_photo.jpg", "wb") as new_file:
            new_file.write(downloaded_file)
        if color_mode=="color":
            cv2r("received_photo.jpg")
        else:
            ai("received_photo.jpg")
        with open("snd.jpg", "rb") as photo:
            bot.send_photo(chat_id, photo, caption="Восстановленное фото")

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке фото",
            reply_markup=create_main_menu()
        )

@bot.message_handler(func=lambda message: True)
def handle_other_messages(message):
    bot.send_message(
        message.chat.id,
        "Пожалуйста, используйте кнопки меню для выбора режима обработки фото или отправьте фото.",
        reply_markup=create_main_menu()
    )

def main():
    print("Бот запущен...")
    bot.infinity_polling()

if __name__ == "__main__":
    main()
