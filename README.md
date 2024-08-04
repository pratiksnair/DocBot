# DocBot

**DocBot** is an interactive chatbot designed to assist users with various medical consultations and general inquiries. It leverages natural language processing and machine learning to understand user inputs and provide relevant responses.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)
8. [Acknowledgements](#acknowledgements)

## Introduction

DocBot is designed to facilitate medical consultations by analyzing user inputs and providing appropriate responses based on predefined intents. The bot can handle various topics including symptoms, prevention methods, and general information.

## Features

- **Natural Language Understanding:** Utilizes NLP techniques for understanding and processing user inputs.
- **Predefined Intents:** Responds to common medical queries based on a set of predefined intents.
- **Streamlit Integration:** Provides a user-friendly web interface for interacting with the chatbot.
- **Customizable Responses:** Easily modify responses and intents to fit specific needs.

## Installation

To install and run DocBot, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/pratiksnair/DocBot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd DocBot
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary files:
   - `intents.json` – Contains the intents and their associated patterns and responses.
   - `words.pkl` – Preprocessed list of unique words.
   - `classes.pkl` – Preprocessed list of intent classes.
   - `chatbot_model.h5` – Pretrained model file.

## Usage

To start the chatbot, run the following command:

```bash
streamlit run chatbot.py
```

This will open a Streamlit application where you can interact with DocBot.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -am 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Submit a pull request.

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact:

- **Pratik S Nair** - [@pratiksnair](https://github.com/pratiksnair)
- **Akash Choudhary** - [@iakashchoudhary](https://github.com/iakashchoudhary)

## Acknowledgements

- Special thanks to the open-source community for providing the tools and libraries used in this project.
- Acknowledgements to contributors and supporters.
