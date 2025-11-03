import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from flask import Flask, render_template, request, jsonify

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    image_data = file.read()
    
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
        gender_neutral_caption=True,
    )

    results = []
    print("Image analysis results:")
    print(" Caption:")
    if result.caption is not None:
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

    print(" Read:")
    if result.read is not None:
        for block in result.read.blocks:
            for line in block.lines:
                print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
                results.append(line.text)
                for word in line.words:
                    print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)