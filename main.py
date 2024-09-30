import telebot
import config
from finding import find_info


bot = telebot.TeleBot(config.TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Отправьте фото цветка.")


@bot.message_handler(content_types=['photo'])
def photo_id(message):
    try:

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = file_info.file_path;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "Starting the search...")

        # Вызываем функцию распознавания цветка
        sort, rec = find_info(src)


        text = "Beautiful " + sort + '!'
        bot.send_message(message.chat.id, text)

        text = 'Care recomendations: ' + rec
        bot.send_message(message.chat.id, text)

    except Exception as e:
        bot.reply_to(message, e)


# RUN
bot.polling(none_stop=True)