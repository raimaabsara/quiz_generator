import random
from flask import Flask, request, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# Set up the Hugging Face pipeline for question generation
# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-base-qa-qg-hl",  # Using a pre-trained model for Q&A generation
    device=0 if device.type == "cuda" else -1
)


def generate_questions_with_answers(document, num_questions=5):
    """
    Function to generate a specified number of question-answer pairs
    from the given document.
    """
    generated_pairs = []
    chunk_size = 256  # Limit the chunk size to match the model's token input limit

    # Split the document into smaller chunks to avoid exceeding model input limits
    chunks = [document[i : i + chunk_size] for i in range(0, len(document), chunk_size)]

    # Keep generating question-answer pairs until we have the required number
    while len(generated_pairs) < num_questions:
        chunk = random.choice(chunks)  # Pick a random chunk for variety

        # Generate a question from the selected chunk
        qg_input = f"generate question: {chunk}"
        generated_text = qg_pipeline(qg_input, max_length=64, num_return_sequences=1)
        question = generated_text[0]["generated_text"].strip()

        # Generate an answer based on the question and the chunk
        qa_input = f"answer: {question} context: {chunk}"
        generated_text = qg_pipeline(qa_input, max_length=64, num_return_sequences=1)
        answer = generated_text[0]["generated_text"].strip()

        # Avoid duplicate question-answer pairs
        if (question, answer) not in generated_pairs:
            generated_pairs.append((question, answer))

    return generated_pairs


@app.route("/")
def home():
    # Render the main page where users can input text
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Endpoint to generate a quiz based on user-provided text.
    Takes input from the form and returns the generated questions and answers.
    """
    document = request.form["text"]  # User-provided text
    num_questions = int(request.form.get("num_questions", 5))  # Number of questions to generate

    if not document:
        return "Please provide some text!", 400  # Handle empty input gracefully

    try:
        # Call the function to generate question-answer pairs
        qa_pairs = generate_questions_with_answers(document, num_questions)
    except Exception as e:
        # Catch and display any errors that occur during question generation
        return f"Error generating questions: {str(e)}", 500

    # Format the generated questions and answers for the result page
    questions = [{"question": pair[0], "answer": pair[1]} for pair in qa_pairs]

    return render_template("result.html", questions=questions)


if __name__ == "__main__":
    # Run the Flask app in debug mode for development
    app.run(debug=True)
