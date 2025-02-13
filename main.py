import logging
import requests
import time
import json
import os
import chromadb
import PyPDF2
from better_profanity import profanity
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = "7652526472:AAHjimlKRI0S3peNiXHwWA7EDxDUmbEMS1A"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
DOWNLOAD_PATH = "downloads" 

DB_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
chat_collection = chroma_client.get_or_create_collection(name="chat_history")
docs_collection = chroma_client.get_or_create_collection(name="documents")

def get_updates(offset=None):
    url = TELEGRAM_API_URL + "getUpdates"
    params = {"timeout": 30, "offset": offset}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching updates from Telegram API: {e}")
        return {}

def send_telegram_message(chat_id, message):
    url = TELEGRAM_API_URL + "sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info(f"✅ Sent message to {chat_id}: {message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send message to {chat_id}: {e}")

def save_chat_history(chat_id, messages):
    chat_data = json.dumps([msg.model_dump() for msg in messages])  
    chat_collection.upsert(documents=[chat_data], ids=[str(chat_id)])

def load_chat_history(chat_id):
    try:
        result = chat_collection.get(ids=[str(chat_id)])
        if result["documents"]:
            return [ChatMessage(**msg) for msg in json.loads(result["documents"][0])]
    except Exception as e:
        logging.error(f"Error loading chat history for {chat_id}: {e}")
    return []

def check_for_profanity(text):
    return profanity.contains_profanity(text)

def get_stored_documents(chat_id):
    try:
        result = docs_collection.get(ids=[str(chat_id)])
        if result["documents"]:
            return result["documents"][0]
    except Exception as e:
        logging.error(f"Error retrieving documents for chat {chat_id}: {e}")
    return ""

def generate_ai_response(chat_id, user_text):
    if check_for_profanity(user_text):
        send_telegram_message(chat_id, "⚠️ Your message contains inappropriate language. Please rephrase it.")
        return ""
    
    model_name = "llama3.2"
    llm = Ollama(model=model_name, request_timeout=120.0)

    chat_history = load_chat_history(chat_id)
    stored_documents = get_stored_documents(chat_id)

    context = f"Chat history:\n{chat_history}\n\nDocuments:\n{stored_documents}"

    chat_history.append(ChatMessage(role="user", content=user_text))
    
    response = ""
    try:
        for message_chunk in llm.stream_chat([ChatMessage(role="system", content=context)] + chat_history):
            response += message_chunk.delta
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        response = "Sorry, I encountered an error. Please try again later."

    chat_history.append(ChatMessage(role="assistant", content=response))
    save_chat_history(chat_id, chat_history)
    return response

def process_messages():
    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if "result" in updates:
            for update in updates["result"]:
                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    user_text = update["message"].get("text", "")

                    logging.info(f"📩 Received message from {chat_id}: {user_text}")
                    ai_response = generate_ai_response(chat_id, user_text)
                    send_telegram_message(chat_id, ai_response)
                    last_update_id = update["update_id"] + 1
        time.sleep(1)

if __name__ == "__main__":
    logging.info("🤖 Bot is running...")
    process_messages()