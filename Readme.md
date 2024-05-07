**Banking Chatbot**

*Overview:*

This submission folder contains code for a conversational chatbot designed to facilitate banking interactions, with a focus on providing a user-friendly experience. The chatbot enables users to perform tasks such as checking account details, transferring money, and engaging in casual conversation.

*Getting Started:*

1. **Run the Application:**
   ```bash
   streamlit run chatbot_app.py
   ```

2. **Speech Input (Optional):**
   - Click the microphone icon in the sidebar to enable speech input.
   - Speak into the microphone when prompted.
   - The chatbot will process your spoken input and provide responses.

3. **Interact with the Chatbot:**
   - Access the chatbot interface through a web browser.
   - Register:
      - Enter your name in the provided text input.
      - Click the "Register" button.
      - Your unique authorization code will be displayed upon successful registration.
   - Check Account Balance:
      - Enter the account number in the "Enter the account number:" text input.
      - Click the "Get balance" button.
      - The account balance will be displayed.
   - Check Dues:
      - Enter the account number in the "Enter the account number:" text input.
      - Click the "Get due" button.
      - The dues will be displayed.
   - Transfer Money:
      - Click the "Ask" button to initiate a money transfer.
      - Follow the prompts to enter recipient's account number, your account number, and the amount to transfer.
      - Click the "Complete Transfer" button to finalize the transaction.

**Note:** 
   - Assign your OpenAI key in your environment variables for the ChatOpenAI chatbot.

*Folder Structure:*

- `chatbot_app.py`: Main application script.
- `database.json`: JSON file storing user account details.
- `Report.pdf`: Detailed report regarding the code and it's functionality
- `Execution_functions`: A ipnyb file to check the working of functions individually

**Team Name:** 
- Phineas and Ferb

**Team Members:** 
- R Adarsh(12141350) 
- R Chiranjeevi Srinivas (12141290)
