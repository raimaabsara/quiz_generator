from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize the pipeline with the correct model
generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    text = request.form["text"]
    if not text:
        return "Please provide some text!", 400

    prompt = f"Generate a multiple-choice question based on: {text}"
    result = generator(prompt, max_length=256, do_sample=True, temperature=0.7, top_k=50)
    question = result[0]["generated_text"]

    return render_template("result.html", question=question)

if __name__ == "__main__":
    app.run(debug=True)
