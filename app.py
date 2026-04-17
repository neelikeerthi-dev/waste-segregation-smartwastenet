from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("models/waste_model.h5")

UPLOAD_FOLDER = "static/uploads"

# Create folder automatically if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    return "Inorganic Waste ♻️" if pred > 0.7 else "Organic Waste 🌱"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        result = predict_label(filepath)
        return render_template("result.html",prediction=result,image_path="static/uploads/" + file.filename)
    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)
