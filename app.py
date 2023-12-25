from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/qa', methods=['POST'])
def question_answering():
    data = request.json
    question = data.get("question")
    context = data.get("context")
    result = pipe(question=question, context=context)
    return jsonify(result)

