from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the tokenizer and model from the pre-trained DialoGPT-medium
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Create a Flask application instance
app = Flask(__name__)

# Define the route for the homepage
@app.route("/")
def index():
    # Render the chat.html template when the homepage is accessed
    return render_template("chat.html")

# Define the route for the chat function, supporting both GET and POST methods
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get the user message from the form
    msg = request.form["msg"]
    input = msg
    # Get the chat response and return it
    return get_Chat_response(input)

# Function to generate a chat response
def get_Chat_response(text):
    # Chat for 5 lines
    for step in range(5):
        # Encode the user input and add the end-of-sequence token, returning a tensor
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # Append the new user input tokens to the chat history, if any
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # Generate a response from the model, limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode and return the last output tokens from the bot, skipping special tokens
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# Run the Flask application
if __name__ == "__main__":
    app.run()