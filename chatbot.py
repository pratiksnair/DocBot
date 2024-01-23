import random  # Importing the random module for generating responses with a random element
import json  # Importing the JSON module for handling data in JavaScript Object Notation
import pickle  # Importing the pickle module for serializing and deserializing Python objects
import numpy as np  # Importing the NumPy library for efficient numerical operations
import nltk  # Importing the Natural Language Toolkit (nltk) for various natural language processing tasks

from nltk.stem import WordNetLemmatizer  # Importing the WordNet lemmatizer from nltk for word normalization
from keras.models import load_model  # Importing the load_model function from Keras for loading a pre-trained machine learning model

import time # Importing the time module for simulation
import streamlit as st # Importing the Streamlit library for creating web applications

# Importing the st_message function from the streamlit_chat module for displaying messages in Streamlit
from streamlit_chat import message as st_message

# Importing message types (SystemMessage, HumanMessage, AIMessage) from the langchain.schema module
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from a JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed words, classes, and the trained model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# Define a function to clean up a sentence by tokenizing and lemmatizing
def clean_up_sentence(sentence):
    # Tokenize the input sentence into a list of words using nltk.word_tokenize
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word in the tokenized sentence using the WordNet lemmatizer
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    # Return the list of lemmatized words
    return sentence_words

# Define a function to convert a sentence into a bag of words representation
def bag_of_words(sentence):
    # Obtain a list of lemmatized words by calling the clean_up_sentence function
    sentence_words = clean_up_sentence(sentence)
    # Initialize a list (bag) with zeros, with the length equal to the total number of unique words in the model
    bag = [0] * len(words)
    # Iterate through each word in the lemmatized sentence
    for w in sentence_words:
        # Check if the word is in the preprocessed list of unique words (words)
        for i, word in enumerate(words):
            # If the word is found in the list, set the corresponding element in the bag to 1
            if word == w:
                bag[i] = 1
    # Convert the bag list into a NumPy array and return it
    return np.array(bag)

# Define a function to predict the class of a given sentence
def predict_class(sentence):
    # Obtain the bag of words representation for the input sentence
    bow = bag_of_words(sentence)
    # Use the model to predict the probability distribution for the given bag of words
    res = model.predict(np.array([bow]))[0]
    # Set an error threshold to filter out predictions with low confidence
    ERROR_THRESHOLD = 0.25
    # Create a list of intent-index and probability pairs that exceed the error threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort the results in descending order based on the probability
    results.sort(key=lambda x: x[1], reverse=True)
    # Create a list of dictionaries containing the predicted intent and its probability
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    # Return the list of predicted intents and probabilities
    return return_list

# Define a function to get a response based on predicted intents
def get_response(intents_list, intents_json):
    if not intents_list:
        # Return a random response when the prediction list is empty
        fallback_responses = ["I'm sorry, I didn't understand that.",
                              "Could you please rephrase that?",
                              "I'm not sure what you mean.",
                              "I currently don't have enough knowledge on that topic.",
                              "I'm having trouble understanding. Can you please rephrase your question?",
                              "I'm not trained on that content or knowledge. Is there anything else I can help you with?"]
        return random.choice(fallback_responses)
    
    # Extract the predicted intent from the intents_list
    tag = intents_list[0]['intent']
    # Get the list of intents from the intents_json
    list_of_intents = intents_json['intents']
    # Iterate through each intent in the list_of_intents
    for i in list_of_intents:
        # Check if the tag of the current intent matches the predicted intent
        if i['tag'] == tag:
            # Randomly choose a response from the responses associated with the matched intent
            result = random.choice(i['responses'])
            # Break out of the loop once a response is chosen
            break
    # Return the selected response
    return result

# Streamlit app
def main():
    # Set up Streamlit page configuration with a title and icon
    st.set_page_config(
        page_title="DocBot: One-Step Medical Consultation",
        page_icon="📜",
    )

    # Set the title of the page
    st.title("🤖DocBot")

    # Initialize session state to keep track of whether the greeting has been displayed
    if "greeting_displayed" not in st.session_state:
        st.session_state.greeting_displayed = False

    # Display greeting message from DocBot if it hasn't been displayed yet
    if not st.session_state.greeting_displayed:
        DocBot_greeting = "Hello! I'm DocBot. How can I assist you today?"
        st.info(DocBot_greeting)
        st.session_state.greeting_displayed = True

    # Display a disclaimer message about using DocBot responsibly
    st.info("Please use DocBot responsibly. Do not misuse it for inappropriate or harmful purposes.")

    # Initialize session state to keep track of chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a very helpful assistant!")
        ]

    # Get user input through a chat input box
    prompt = st.chat_input("Message DocBot", key="prompt")

    # If the user has entered a message, process it and generate an assistant response
    if prompt:
        # Add the user's message to the session state
        st.session_state.messages.append(HumanMessage(content=prompt))
        # Use a function 'predict_class' to predict the intent of the user's message
        ints = predict_class(prompt)
        # Use a function 'get_response' to generate an assistant response based on the predicted intent
        assistant_response = get_response(ints, intents)
        # Add the assistant's response to the session state
        st.session_state.messages.append(AIMessage(content=assistant_response))

    # Retrieve the messages from the session state
    messages = st.session_state.get('messages', [])

    # Display the conversation history, alternating between user and bot messages
    for i, msg in enumerate(messages[1:]):  # Start from index 1 to skip the system message
        if i % 2 == 0:
            # Display user's message
            st_message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            # Display bot's message
            st_message(msg.content, is_user=False, key=str(i) + '_bot')

# Run the Streamlit app if the script is executed
if __name__ == "__main__":
    main()