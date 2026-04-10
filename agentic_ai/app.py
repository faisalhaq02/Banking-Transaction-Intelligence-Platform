from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from agentic_ai.agent import run_agent

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.get_json(force=True) or {}
        user_query = payload.get("message", "")
        response = run_agent(user_query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)