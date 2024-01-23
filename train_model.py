import random  # Importing the random module for generating random values and shuffling data
import json  # Importing the JSON module for working with JSON data
import pickle  # Importing the pickle module for serializing and deserializing Python objects
import numpy as np  # Importing the NumPy library for numerical operations
import tensorflow as tf  # Importing TensorFlow, an open-source machine learning framework

import nltk  # Importing the Natural Language Toolkit (nltk) for natural language processing
from nltk.stem import WordNetLemmatizer  # Importing the WordNet lemmatizer from nltk for word normalization

from keras.callbacks import TensorBoard  # Importing the TensorBoard callback from Keras for model visualization

import os  # Importing the os module for interacting with the operating system
import warnings  # Importing the warnings module for controlling warning messages

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from a JSON file
intents = json.loads(open('intents.json').read())

# Initialize lists to store words, classes, and documents
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Loop through intents and patterns to preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern into a list of words
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Add (wordList, intent['tag']) tuple to documents
        documents.append((wordList, intent['tag']))
        # Add the intent tag to the classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]

# Remove duplicate words and sort them
words = sorted(set(words))

# Sort the classes
classes = sorted(set(classes))

# Save words and classes as pickle files for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize lists for training data
training = []
outputEmpty = [0] * len(classes)

# Create the training data by converting word patterns into a bag of words representation and one-hot encoded output
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        # Create bag-of-words representation
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    # Create the output row with 0s and 1 at the corresponding class index
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    # Combine bag of words representation and output row
    training.append(bag + outputRow)

# Shuffle the training data
# random.shuffle(training)

# Convert training data to a numpy array
training = np.array(training)

# Separate input features (X) and output labels (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Create a sequential neural network model using TensorFlow and Keras
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configure the model with stochastic gradient descent (SGD) optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model with categorical crossentropy loss and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Set up TensorBoard callback
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model on the training data
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1, callbacks=[tensorboard_callback])

# Save the trained model
model.save('chatbot_model.keras', hist)
with open('training_history.pkl', 'wb') as file:
    pickle.dump(hist.history, file)

print('Model Trained Successfully!')