from flask import Flask, render_template, jsonify, request, Response, session, redirect, url_for
from werkzeug.utils import secure_filename
from router import graph
import os
import time

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

agent_labels = {
        "home_inspection_agent": "🛠️ Home Inspection Agent",
        "faq_agent": "📘 FAQ Agent",
        "clarify": "❓ Clarification",
        "unknown": "🤖 Unknown Agent"
    }

@app.route("/", methods=["GET"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["message"]
    image_file = request.files.get("image")

    image_path = None
    if image_file and image_file.filename:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(image_path)

    session["chat_history"].append({"role": "user", "text": user_input})

    memory = []
    for msg in session["chat_history"]:
        if msg["role"] == "user":
            memory.append(f"User: {msg['text']}")
        elif msg["role"] == "assistant":
            memory.append(f"Agent: {msg['text']}")

    result = graph.invoke({
        "input": user_input,
        "image_path": image_path,
        "memory": memory
    })

    final_response = result["agent_response"]
    route_used = result.get("route", "unknown")    

    label = agent_labels.get(route_used, "🤖 Unknown Agent")
    response_with_label = f"{label}:\n{final_response}"

    session["chat_history"].append({"role": "assistant", "text": response_with_label})

    return jsonify({"response": response_with_label})

@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run()
