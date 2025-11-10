import telebot 

from app.main import run_pipeline


bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я твой бот, которые отвечает на faq!")


@bot.message_handler(commands=['que'])
def que_user(message):
    bot.send_message(message.chat.id, 'Введите вопрос, который вас интересует')

    bot.register_next_step_handler(message, process_que)

def process_que(message):
    user_text = message.text

    result = run_pipeline(user_text=user_text)

    bot.send_message(message.chat.id, result)

bot.polling()