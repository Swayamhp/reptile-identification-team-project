from flask import Flask, request, jsonify,render_template
import numpy as np
import tensorflow as tf
import os
from PIL import Image

app = Flask(__name__)

# ✅ Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "full_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Define class labels
labels = {
    0: 'Chameleon', 
    1: 'Crocodile', 
    2: 'Frog', 
    3: 'Gecko', 
    4: 'Iguana', 
    5: 'Lizard', 
    6: 'Salamander', 
    7: 'Snake', 
    8: 'Toad', 
    9: 'Turtle'
}

# ✅ Define route to check if API is working
@app.route("/")
def home():
    return render_template("index.html")  # Load the HTML page

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    try:
        # ✅ Load and preprocess the image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))  # Resize to model input size
        img_array = np.array(image) / 255.0  # Normalize
        img_batch = np.expand_dims(img_array, axis=0)  # Expand dims for batch

        # ✅ Make a prediction
        predictions = model.predict(img_batch)
        predicted_class_index = np.argmax(predictions)
        predicted_label = labels[predicted_class_index]

        # ✅ Return JSON response
        return jsonify({"prediction": predicted_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns a PORT dynamically
    app.run(host="0.0.0.0", port=port)





