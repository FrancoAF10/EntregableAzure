import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Cargar variables del archivo .env
load_dotenv()

language_key = os.getenv("LANGUAGE_KEY")
language_endpoint = os.getenv("LANGUAGE_ENDPOINT")

if not language_key or not language_endpoint:
    raise ValueError("Faltan las variables LANGUAGE_KEY o LANGUAGE_ENDPOINT en .env")

# Configurar cliente de Azure
credential = AzureKeyCredential(language_key)
client = TextAnalyticsClient(endpoint=language_endpoint, credential=credential)

# Inicializar la app Flask
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("analize.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Debe ingresar un texto"}), 400

    try:
        result = client.analyze_sentiment([text], show_opinion_mining=False)
        doc = result[0]
        return jsonify({
            "sentiment": doc.sentiment,
            "confidence": {
                "positive": doc.confidence_scores.positive,
                "neutral": doc.confidence_scores.neutral,
                "negative": doc.confidence_scores.negative
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
