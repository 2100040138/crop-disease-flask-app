import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

app = Flask(__name__)
CORS(app)

# ========= CONFIGURATION ==========
CROP_CSV_PATH = r"C:\Users\LENOVO\archive\Crop_recommendation.csv"
TRAIN_DIR = r"C:\Users\LENOVO\plant_disease_dataset\Train"
VAL_DIR = r"C:\Users\LENOVO\plant_disease_dataset\Validation"

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ========= TRAIN MODELS ==========
def train_crop_model():
    df = pd.read_csv(CROP_CSV_PATH)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'crop_model.pkl')

def train_disease_model():
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(TRAIN_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')
    val_data = datagen.flow_from_directory(VAL_DIR, target_size=(150,150), batch_size=32, class_mode='categorical')

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_data.class_indices), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save('plant_disease_model.keras')

    with open('disease_labels.txt', 'w') as f:
        for label in train_data.class_indices:
            f.write(label + '\n')

# ========= LOAD MODELS ==========
def load_models():
    global crop_model, disease_model, disease_labels
    crop_model = joblib.load('crop_model.pkl')
    disease_model = load_model('plant_disease_model.keras')
    with open('disease_labels.txt') as f:
        disease_labels = [line.strip() for line in f.readlines()]

# ========= HELPER FUNCTIONS ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========= HTML TEMPLATE ==========
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
  <title>🌾 Crop & Disease Predictor</title>
  <style>
    body { font-family: Arial; background-color: #f8f9fa; padding: 20px; }
    h1 { color: green; }
    form { background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px #ccc; margin-bottom: 30px; }
    input, button { width: 100%; padding: 10px; margin: 8px 0; }
    button { background-color: green; color: white; border: none; }
    .result { font-weight: bold; color: #222; background: #e2f0d9; padding: 15px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>🌿 Smart Agriculture Dashboard</h1>

  <form method="POST" action="/predict_crop">
    <h2>Crop Prediction</h2>
    <input name="N" placeholder="Nitrogen (N)" required>
    <input name="P" placeholder="Phosphorus (P)" required>
    <input name="K" placeholder="Potassium (K)" required>
    <input name="temperature" placeholder="Temperature" required>
    <input name="humidity" placeholder="Humidity" required>
    <input name="ph" placeholder="pH Level" required>
    <input name="rainfall" placeholder="Rainfall" required>
    <button type="submit">Predict Crop</button>
  </form>

  <form method="POST" action="/predict_disease" enctype="multipart/form-data">
    <h2>Plant Disease Detection</h2>
    <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
    <button type="submit">Predict Disease</button>
  </form>

  {% if result %}
    <div class="result">{{ result }}</div>
  {% endif %}
</body>
</html>
'''

# ========= ROUTES ==========
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        prediction = crop_model.predict([features])[0]
        return render_template_string(HTML_TEMPLATE, result=f"✅ Recommended Crop: {prediction}")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=f"❌ Error: {e}")

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, result="❌ No file uploaded")
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = disease_model.predict(img_array)
            index = np.argmax(predictions[0])
            confidence = predictions[0][index]
            label = disease_labels[index]

            return render_template_string(HTML_TEMPLATE, result=f"🦠 Detected Disease: {label} (Confidence: {confidence:.2f})")
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, result=f"❌ Prediction Error: {e}")
        finally:
            os.remove(filepath)
    else:
        return render_template_string(HTML_TEMPLATE, result="❌ Invalid file type")

# ========= MAIN ==========
if __name__ == '__main__':
    print("📊 Training crop model...")
    train_crop_model()

    print("🌱 Training disease model...")
    train_disease_model()

    print("🔄 Loading models...")
    load_models()

    print("🚀 Starting app...")
    # Very Important for Hosting
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
