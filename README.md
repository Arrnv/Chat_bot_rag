# Conversational Q&A Chatbot

This project is a conversational Q&A chatbot application built with Streamlit and LangChain, leveraging Google Generative AI for natural language processing. The chatbot uses PDF documents as a knowledge base to answer user queries.

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)

## Features
- Conversational interface with chat history
- Uses LangChain for document processing and question-answering
- Leverages Google Generative AI for natural language understanding

## Requirements
- Python 3.8 or later
- Streamlit
- LangChain
- PyPDF2
- Google Generative AI SDK

## Setup

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/chatbot-project.git
cd chatbot-project
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
conda create --name venv
conda activate venv
```
Then install the requirements.txt
```bash
pip install -r requirements.txt
```
### 3. Set Up API Key
Replace the api key in .env file 


### 4. Prepare the PDF Document
Ensure that your knowledge base PDF is named Corpus.pdf and placed in the root directory of the project.

### 5. Run the Application
Start the Streamlit application:

```bash
streamlit run app.py
```
