from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

model = OllamaLLM(model="llama3")

template = """
Answer the question below:
The conversation history: {context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

context = ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    print('this is ask')
    global context
    user_input = request.json.get("question")
    result = chain.invoke({"context": context, "question": user_input})
    context += f"\nUser: {user_input}\nAI: {result}"
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)
