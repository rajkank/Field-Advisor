# Importing essential libraries and modules

from flask import Flask, render_template, request, jsonify, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Preload fertilizer data once and expose crop list to templates
fertilizer_df = pd.read_csv('Data/fertilizer.csv')
fertilizer_crops = sorted(fertilizer_df['Crop'].unique())


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity or None on error
    """
    api_key = (config.weather_api_key or "").strip()
    if not api_key:
        return None

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    try:
        response = requests.get(
            base_url,
            params={"appid": api_key, "q": city_name, "units": "metric"},
            timeout=10,
        )
        x = response.json()
    except (requests.RequestException, ValueError):
        return None

    # API returns cod as 200 (int) for success, "404" (str) for not found, or other error codes
    cod = x.get("cod")
    if cod == "404" or cod == 404 or cod is None:
        return None
    if cod != 200 and cod != "200":
        # Log API error (401=bad key, 429=rate limit, etc.) for debugging
        try:
            app.logger.warning("OpenWeatherMap API error: cod=%s message=%s", cod, x.get("message"))
        except Exception:
            pass
        return None

    # Success response must have "main" with "temp" and "humidity"
    if "main" not in x or not isinstance(x["main"], dict):
        return None
    y = x["main"]
    if "temp" not in y or "humidity" not in y:
        return None

    # API with units=metric returns temp in Celsius already
    temp = y.get("temp")
    if temp is None:
        return None
    temperature = round(float(temp), 2)
    humidity = int(y["humidity"])
    return temperature, humidity


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)


@app.context_processor
def inject_common_template_data():
    """
    Inject commonly used data into all templates.
    Currently provides:
    - fertilizer_crops: list of crop names available in fertilizer.csv
    """
    return dict(fertilizer_crops=fertilizer_crops)

# render home page


@ app.route('/')
def home():
    title = 'Field Advisor - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Field Advisor - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Field Advisor - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Field Advisor - Crop Recommendation'
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            if is_ajax:
                return jsonify({'success': True, 'prediction': final_prediction})
            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Could not fetch weather for this location. Please try another city.'})
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Field Advisor - Fertilizer Suggestion'
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    # Look up the recommended N, P, K values for the selected crop
    crop_row = fertilizer_df[fertilizer_df['Crop'] == crop_name].iloc[0]
    nr = crop_row['N']
    pr = crop_row['P']
    kr = crop_row['K']

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))
    if is_ajax:
        return jsonify({'success': True, 'recommendation': str(fertilizer_dic[key])})

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Field Advisor - Disease Detection'
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        if 'file' not in request.files:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Please upload an image.'})
            return redirect(request.url)

        file = request.files.get('file')
        if not file:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Please upload an image.'})
            return render_template('disease.html', title=title)

        try:
            img = file.read()
            prediction_key = predict_image(img)
            prediction_html = str(disease_dic[prediction_key])

            if is_ajax:
                return jsonify({'success': True, 'prediction': prediction_html})

            prediction = Markup(prediction_html)
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception:
            if is_ajax:
                return jsonify({'success': False, 'error': 'Could not process the image. Please try another one.'})
            return render_template('disease.html', title=title)

    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
