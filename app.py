from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    text = request.form["text"]
    return f"You entered: {text}"

if __name__ == "__main__":
    app.run(debug=True)
