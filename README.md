# Quiz Generator (In Progress)

This is a Flask-based web application that uses Hugging Face Transformers to generate question-and-answer pairs from user-provided text.

## Features
- Generates up to 5 question-and-answer pairs based on user input.
- Displays the answers with a "Reveal Answer" button for an interactive experience.
- Powered by the `valhalla/t5-base-qa-qg-hl` model from Hugging Face.

## Current Status
- **Work in Progress**: 
  - The app generates question-and-answer pairs but does not yet support multiple-choice questions.
  - It may experience delays or issues with large text inputs.
  - Further optimization and UI improvements are planned.

## Requirements
- Python 3.10+
- Flask
- Hugging Face Transformers
- PyTorch

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/raimaabsara/quiz_generator
   ```
2. Navigate to the project directory:
   ```bash
   cd quiz_generator
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://127.0.0.1:5000`.
3. Enter your text and specify the number of questions (maximum 5) to generate a quiz.

