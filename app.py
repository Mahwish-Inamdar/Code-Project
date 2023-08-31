import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img
from keras.utils import img_to_array

app = Flask(__name__)
model = load_model("sunflower_cnn_model.h5")
classes = ["Healthy", "Alternaria Leaf Blight"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the uploaded image
    file = request.files["file"]
    if not file:
        return jsonify({"error": "Please select an image file."})

    # Save the uploaded image to a temporary file
    temp_path = "temp.jpg"
    file.save(temp_path)

    # Preprocess the image for prediction
    img = load_img(temp_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0
    x = x.reshape((1, 224, 224, 3))

    # Make predictions on the image
    preds = model.predict(x)[0]
    class_idx = preds.argmax()
    prediction = classes[class_idx]
    score = preds[class_idx]

    # Create a response JSON object with the prediction and image path
    response = {"prediction": f"{prediction} ({score:.2f})", "image_path": f"static/{file.filename}"}

    # Save the uploaded image to the static folder for display on the front-end
    file_path = os.path.join("static", file.filename)
    os.rename(temp_path, file_path)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

