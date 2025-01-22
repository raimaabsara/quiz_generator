import random
from flask import Flask, request, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# Initialize the Hugging Face pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-base-qa-qg-hl",  # Adjust the model to your preferred one
    device=0 if device.type == "cuda" else -1
)


def generate_questions_with_answers(document, num_questions=5):
    """Generate question-answer pairs from the document."""
    generated_pairs = []
    chunk_size = 256  # Token limit for the model input

    # Split the document into manageable chunks
    chunks = [document[i : i + chunk_size] for i in range(0, len(document), chunk_size)]

    # Generate questions and answers until reaching the desired number
    while len(generated_pairs) < num_questions:
        chunk = random.choice(chunks)  # Select a random chunk

        # Generate a question
        qg_input = f"generate question: {chunk}"
        generated_text = qg_pipeline(qg_input, max_length=64, num_return_sequences=1)
        question = generated_text[0]["generated_text"].strip()

        # Generate the answer
        qa_input = f"answer: {question} context: {chunk}"
        generated_text = qg_pipeline(qa_input, max_length=64, num_return_sequences=1)
        answer = generated_text[0]["generated_text"].strip()

        # Ensure uniqueness
        if (question, answer) not in generated_pairs:
            generated_pairs.append((question, answer))

    return generated_pairs


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate the quiz from user input."""
    document = request.form["text"]
    num_questions = int(request.form.get("num_questions", 5))

    if not document:
        return "Please provide some text!", 400

    try:
        qa_pairs = generate_questions_with_answers(document, num_questions)
    except Exception as e:
        return f"Error generating questions: {str(e)}", 500

    # Prepare the data for rendering
    questions = [{"question": pair[0], "answer": pair[1]} for pair in qa_pairs]

    return render_template("result.html", questions=questions)


if __name__ == "__main__":
    app.run(debug=True)
