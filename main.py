import logging
import requests
import time
import json
import os
import threading
import chromadb
import PyPDF2
import streamlit as st
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

LOCK_FILE = "bot.lock"


def ensure_bot_not_locked():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logging.info("🔄 Removed old lock file.")


def get_updates(offset=None):
    url = TELEGRAM_API_URL + "getUpdates"
    params = {"timeout": 30, "offset": offset}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error fetching updates from Telegram API: {e}")
        return {}


def send_telegram_message(chat_id, message):
    url = TELEGRAM_API_URL + "sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info(f"✅ Sent message to {chat_id}: {message}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send message to {chat_id}: {e}")


def check_for_profanity(text):
    return profanity.contains_profanity(text)


def process_telegram_document(chat_id, file_id, file_name):
    file_info_url = f"{TELEGRAM_API_URL}getFile?file_id={file_id}"
    response = requests.get(file_info_url)
    if response.status_code != 200:
        logging.error(f"❌ Failed to get file info: {response.text}")
        return

    file_path = response.json()["result"]["file_path"]
    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    file_extension = os.path.splitext(file_name)[1].lower()
    local_file_path = os.path.join(DOWNLOAD_PATH, file_name)

    with open(local_file_path, "wb") as f:
        file_response = requests.get(download_url)
        if file_response.status_code == 200:
            f.write(file_response.content)
            logging.info(f"📂 File downloaded: {local_file_path}")
        else:
            logging.error(f"❌ Error downloading file: {file_response.text}")
            return

    if file_extension == ".pdf":
        text = extract_text_from_pdf(local_file_path)
    elif file_extension == ".txt":
        with open(local_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        send_telegram_message(chat_id, "⚠️ Unsupported file format! Only PDF and TXT are allowed.")
        return

    if check_for_profanity(text):
        send_telegram_message(chat_id, "⚠️ The document contains inappropriate content and was not saved.")
        return

    save_document(chat_id, text)
    send_telegram_message(chat_id, f"✅ Document '{file_name}' successfully processed and saved.")


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    return text


def save_document(chat_id, text):
    try:
        docs_collection.upsert(documents=[text], ids=[str(chat_id)])
        logging.info(f"✅ Document saved for chat {chat_id}.")
    except Exception as e:
        logging.error(f"❌ Error saving document for chat {chat_id}: {e}")


def load_chat_history(chat_id):
    try:
        result = chat_collection.get(ids=[str(chat_id)])
        if result["documents"]:
            return [ChatMessage(**msg) for msg in json.loads(result["documents"][0])]
    except Exception as e:
        logging.error(f"❌ Error loading chat history for {chat_id}: {e}")
    return []


def get_stored_documents(chat_id):
    try:
        result = docs_collection.get(ids=[str(chat_id)])
        if result["documents"]:
            return result["documents"][0]
    except Exception as e:
        logging.error(f"❌ Error retrieving documents for chat {chat_id}: {e}")
    return ""


def generate_ai_response(chat_id, user_text):
    if check_for_profanity(user_text):
        return "⚠️ Your message contains inappropriate language. Please rephrase it."

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
        logging.error(f"❌ Error generating response: {e}")
        response = "Sorry, I encountered an error. Please try again later."

    return response



bot_running = False  

def process_messages():
    global bot_running, last_update_id
    if bot_running:
        logging.info("⚠️ Bot is already running, skipping duplicate start.")
        return
    
    bot_running = True  
    logging.info("✅ Telegram bot started and listening for messages...")
    
    last_update_id = None  

    try:
        while bot_running:
            updates = get_updates(last_update_id) 
            
            if "result" in updates:
                for update in updates["result"]:
                    update_id = update["update_id"]
                    
                    if last_update_id is not None and update_id <= last_update_id:
                        continue
                    
                    last_update_id = update_id  
                    chat_id = update["message"]["chat"]["id"]

                    if "text" in update["message"]:
                        user_text = update["message"]["text"]
                        logging.info(f"📩 Received text from {chat_id}: {user_text}")
                        ai_response = generate_ai_response(chat_id, user_text)
                        send_telegram_message(chat_id, ai_response)

                    elif "document" in update["message"]:
                        file_id = update["message"]["document"]["file_id"]
                        file_name = update["message"]["document"]["file_name"]
                        logging.info(f"📂 Received document from {chat_id}: {file_name}")
                        process_telegram_document(chat_id, file_id, file_name)

            time.sleep(1.5)  

    finally:
        bot_running = False  
        logging.info("🛑 Bot process stopped.")

def process_uploaded_file(uploaded_file, chat_id):
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == ".txt":
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            st.error("⚠️ Unsupported file format! Only PDF and TXT are allowed.")
            return
        
        if check_for_profanity(text):
            st.error("⚠️ The document contains inappropriate content and was not saved.")
            return

        save_document(chat_id, text)
        st.success(f"✅ Document '{uploaded_file.name}' successfully processed and saved.")


def streamlit_ui():
    st.title("🤖 AI Chatbot with Telegram & Document Support")

    uploaded_file = st.file_uploader("📂 Upload a PDF or TXT file:", type=["pdf", "txt"])
    if uploaded_file:
        process_uploaded_file(uploaded_file, "streamlit_user")

    user_input = st.text_input("💬 Enter your message:")
    if st.button("Send"):
        chat_id = "streamlit_user"
        response = generate_ai_response(chat_id, user_input)
        st.write("🤖 Response:", response)


if __name__ == "__main__":
    ensure_bot_not_locked()
    
    st.sidebar.title("⚙️ Options")

    if "bot_thread" not in st.session_state:
        st.session_state.bot_thread = None

    run_telegram_bot = st.sidebar.checkbox("📲 Run Telegram Bot", value=False)
    run_streamlit_ui = st.sidebar.checkbox("🖥 Run Streamlit UI", value=True)

    if run_telegram_bot:
        if st.session_state.bot_thread is None or not st.session_state.bot_thread.is_alive():
            st.session_state.bot_thread = threading.Thread(target=process_messages, daemon=True)
            st.session_state.bot_thread.start()
            logging.info("✅ Telegram bot started.")
    else:
        if st.session_state.bot_thread is not None:
            logging.info("🛑 Stopping Telegram bot...")
            bot_running = False
            st.session_state.bot_thread = None

    if run_streamlit_ui:
        streamlit_ui()
