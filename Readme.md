# Banking Chatbot

## Overview

This repository contains the code for a conversational chatbot designed to facilitate banking interactions, providing a user-friendly experience. The chatbot enables users to check account details, transfer money, and engage in casual conversation.

## Getting Started

### Run the Application

```bash
streamlit run chatbot_app.py
```

### Speech Input (Optional)

1. Click the microphone icon in the sidebar to enable speech input.
2. Speak into the microphone when prompted.
3. The chatbot will process your spoken input and provide responses.

### Interact with the Chatbot

1. **Access the Chatbot Interface:**
   - Open a web browser and access the chatbot interface.

2. **Register:**
   - Enter your name in the provided text input.
   - Click the "Register" button.
   - Your unique authorization code will be displayed upon successful registration.

3. **Check Account Balance:**
   - Enter the account number in the "Enter the account number:" text input.
   - Click the "Get balance" button.
   - The account balance will be displayed.

4. **Check Dues:**
   - Enter the account number in the "Enter the account number:" text input.
   - Click the "Get due" button.
   - The dues will be displayed.

5. **Transfer Money:**
   - Click the "Ask" button to initiate a money transfer.
   - Follow the prompts to enter the recipient's account number, your account number, and the amount to transfer.
   - Click the "Complete Transfer" button to finalize the transaction.

## Note

Assign your OpenAI key in your environment variables for the ChatOpenAI chatbot.

## Folder Structure

- `chatbot_app.py`: Main application script.
- `database.json`: JSON file storing user account details.
- `Report.pdf`: Detailed report regarding the code and its functionality.
- `Execution_functions.ipynb`: Jupyter notebook to check the working of functions individually.


