import streamlit as st
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
import vosk
import pyttsx3
import sounddevice as sd
import numpy as np
import wave
import re
import uuid

# Load Vosk model
vosk_model_path = "C:/Users/hp/Downloads/lm/vosk-model-small-en-us-zamia-0.5"
vosk_model = vosk.Model(vosk_model_path)

# Load conversational chatbot
chatbot = ChatOpenAI(temperature=0.5)

# Load account data from JSON file
json_file_path = "C:/Users/hp/Downloads/lm/database.json"

# Function to record audio to WAV file
def record_audio_to_wav(filename="recorded_audio.wav", duration=5, sample_rate=16000):
    st.info("Recording... Speak into the microphone.")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    st.info("Recording stopped.")
    # Save the recorded audio as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return filename

# Function to recognize speech from WAV file
def recognize_speech_from_wav(wav_filename):
    wf = wave.open(wav_filename, 'rb')
    sample_rate = wf.getframerate()

    recognizer = vosk.KaldiRecognizer(vosk_model, sample_rate)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            pass

    result = recognizer.Result()
    return result

# Function to convert text to speech
def text_to_speech(text):
    st.info("Converting text to speech...")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_banking_chatbot_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chatbot(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# Function to get account details from the database
def get_account_details(account_number):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for account in data['accounts']:
        if account['acc_no'] == int(account_number):
            return {
                "name": account['name'],
                "acc_no": account['acc_no'],
                "balance": account['balance']
            }

    return None

# Function to update account balance in the database
def update_account_balance(account_number, amount):
    # Load account data from JSON file
    with open(json_file_path, "r") as file:
        account_data = json.load(file)

    # Find the account in the data
    for account in account_data["accounts"]:
        if account["acc_no"] == account_number:
            # Update the balance
            account["balance"] += amount
            break

    # Save the updated data back to the JSON file
    with open(json_file_path, "w") as file:
        json.dump(account_data, file, indent=2)


# Function to transfer money between accounts
def transfer_money(sender_acc_no, receiver_acc_no, amount):
    data = read_data_from_json(json_file_path)
    sender_account = None
    receiver_account = None

    # Find sender and receiver accounts
    for account in data['accounts']:
        if account['acc_no'] == int(sender_acc_no):
            sender_account = account
        elif account['acc_no'] == int(receiver_acc_no):
            receiver_account = account

    # Check if sender and receiver accounts exist
    if sender_account is None or receiver_account is None:
        return False, "Sender or receiver account not found."

    # Check if sender has sufficient balance
    if sender_account['balance'] < amount:
        return False, "Insufficient balance."

    # Deduct amount from sender's balance
    sender_account['balance'] -= amount

    # Add amount to receiver's balance
    receiver_account['balance'] += amount

    # Update JSON data
    write_data_to_json(json_file_path, data)

    return True, "Transfer successful. You have transferred " + str(amount) + " to " + str(receiver_acc_no) + "."

# Function to read data from a JSON file
def read_data_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Function to write data to a JSON file
def write_data_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Function to get user intent using the LLM model
def llm_model_for_intent_recognition(test_sentence):
    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained("C:/Users/hp/Downloads/lm/Model_bert_ds_b1")
    tokenizer = BertTokenizer.from_pretrained("C:/Users/hp/Downloads/lm/Model_bert_ds_b1")

    inputs = tokenizer(test_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_labels = torch.argmax(outputs.logits, dim=1)
    return int(predicted_labels)

# Function to get user intent based on LLM predictions
def get_user_intent(text_input):
    # Use LLM for intent recognition
    user_intent = llm_model_for_intent_recognition(text_input)

    # Map the predicted label to your specific intents
    if user_intent == 0:
        return "GET_BALANCE"
    elif user_intent == 1:
        return "GET_DUE"
    elif user_intent == 2:
        return "TRANSFER_MONEY"
    else:
        return "UNKNOWN"

# Function to handle user input based on intent
def handle_user_input():
    if st.session_state.get('values_entered', False): 
        return

    if st.button("Ask"):
        user_input = st.session_state['user_input']
        user_intent = get_user_intent(user_input)

        if user_intent == "TRANSFER_MONEY":
            st.info("Sure, let's initiate a money transfer.")
            st.session_state['expected_input'] = "TRANSFER_MONEY"
            text_to_speech("Sure, let's initiate a money transfer. Please provide the following details:")
            recipient_account = st.text_input("Recipient's account number:")
            sender_account_number = st.text_input("Enter your account number:")
            transfer_amount = st.text_input("Amount to transfer:")
            if st.button("ok"):
                if recipient_account and transfer_amount:
                    success, message = transfer_money(sender_account_number, recipient_account, float(transfer_amount))

                    if success:
                        st.success(message)
                        text_to_speech(f"Transfer successful. {message}")
                    else:
                        st.error(message)
                        text_to_speech(f"Transfer failed. {message}")

                else:
                    st.warning("Please provide both recipient's account number and transfer amount.")
                    text_to_speech("Please provide both recipient's account number and transfer amount.")

        elif user_intent == "GET_BALANCE":
            st.info("Sure, let me check your balance.")
            account_number = st.text_input("Enter the account number:")
            if account_number:
                st.session_state['account_number'] = account_number
                balance = get_account_details(account_number).get('balance', 0)
                st.info(f"Your account balance is {balance}$.")
                text_to_speech(f"Your account balance is {balance}$.")
            else:
                st.warning("Unable to extract account number from the input. Please provide a valid account number.")
                text_to_speech("Sorry, I couldn't understand the account number. Please provide a valid account number.")

        elif user_intent == "GET_DUE":
            st.info("Sure, let me check your dues.")
            account_number = st.text_input("Enter the account number:")
            if st.button("ok"):
                if account_number:
                    st.session_state['account_number'] = account_number
                    dues = get_account_details(account_number).get('dues', 0)
                    st.info(f"Your dues are {dues}$.")
                    text_to_speech(f"Your dues are {dues}$.")
                else:
                    st.warning("Unable to extract account number from the input. Please provide a valid account number.")
                    text_to_speech("Sorry, I couldn't understand the account number. Please provide a valid account number.")

        else:
            st.subheader("Chatbot Response:")
            response = get_banking_chatbot_response(user_input)
            st.write(response)
            text_to_speech(response)

# Streamlit UI
st.set_page_config(page_title="Banking Chatbot")
st.header("Banking Chatbot")
st.image("bank.png")

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="Welcome to the Banking Chatbot. How can I assist you?")
    ]

# Chitchat
st.sidebar.title("User Information")
st.sidebar.info("Click the microphone icon and speak to interact with the chatbot.")
audio_data = st.sidebar.button("🎙️ Ask your queries")
if audio_data:
    wav_filename = record_audio_to_wav()
    text_input = recognize_speech_from_wav(wav_filename)
    st.session_state['user_input'] = text_input
    st.sidebar.text(f"User Input: {text_input}")
else:
    st.session_state['user_input'] = st.text_input("Your Message:")

# Handle user input based on intent
handle_user_input()
