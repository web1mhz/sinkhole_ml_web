from flask import Flask, render_template, request, jsonify

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import rasterio
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

import joblib

app = Flask(__name__)

# Path to your raster file (replace with your actual file path)
raster_file_path = 'static/data/env_raster/dem_84.tif'
raster_file_paths = glob.glob('static/data/env_raster/*.tif')

model_folder= 'static/data/model'

load_model = joblib.load(f'{model_folder}/XGBClassifier.joblib')
# load_model = joblib.load(f'{model_folder}/RandomForestClassifier.joblib')

print(raster_file_paths)

def sns_hangul():

    print("# 한글 깨짐 방지")    
    font_location = 'C:/Windows/Fonts/MALGUN.TTF' # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_coordinates', methods=['POST'])
def process_coordinates():

   

    # Get latitude and longitude from the request
    data = request.get_json()

    lat = float(data['latitude'])
    lon = float(data['longitude']) 
    print("aa",lon, lat)   

    # try:

    # Use coordinates to extract values from the raster file
    # values = extract_values_from_raster(lat, lon)

    for raster_path in raster_file_paths:
        print(raster_path)

    values = [extract_values_from_raster(lat, lon, raster_path) for raster_path in raster_file_paths]

    r_names = [os.path.basename(raster_path).split('.')[0] for raster_path in raster_file_paths]  

    print(values)

    pred_val = model_predict(values, r_names)

    pred_val1 = np.round(pred_val[0], 3)

    str_val1 = str(pred_val1)

    print('tg',str_val1)

    if pred_val1 > 0.75:
        pred_class = '1등급' 
    elif pred_val1 > 0.50:
        pred_class = '2등급'
    elif pred_val1 > 0.25:
        pred_class = '3등급'
    else:
        pred_class = '안전지대'  

    print(pred_class)

    # Create a bar chart and convert it to base64 for embedding in HTML
    chart_img = create_bar_chart(r_names, values)
    chart_base64 = base64.b64encode(chart_img.getvalue()).decode('utf-8')

    return jsonify({'chart_base64': chart_base64, 'results': pred_class, 'pred_val' :  str_val1})

    # except Exception as e:
    #     return jsonify({'error': str(e)})

def extract_values_from_raster(latitude, longitude, raster_path):
    # Open the raster file using rasterio

    with rasterio.open(raster_path) as src:
        
        # print(src.crs)
        data = src.read(1) 

        try:

            x_index, y_index = src.index(longitude,latitude)
            values= data[x_index, y_index]

             # Check if values is not None and has non-zero size
            if values is not None and values.size > 0:
                return values.item(0)
            else:
                # If values is None or has zero size, return a placeholder value (you can modify this as needed)
                return None

        except Exception as e:
            # Handle any exceptions that might occur during the extraction process
            return str(e)

def model_predict(values, r_name):
    print(values, r_name)

    x = np.array([values])
    print(x)
    pred_result = load_model.predict_proba(x)

    pred_result1 = pred_result[:,-1]

    return pred_result1 
   
    

def create_bar_chart(rnames, values): 

    font_location = 'C:/Windows/Fonts/MALGUN.TTF' # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name) 

    rnames_ed =[name.split("_")[0]  for name in rnames]
    

    # Check if values is None
    for r_name in rnames:
        print(r_name)
        if r_name == "hydrgeo_84":            
            values[rnames.index(r_name)] = values[rnames.index(r_name)] * 100
    
    if values is None:
        # Return an empty BytesIO object
        return BytesIO()
    
    # print("c", values)

    # Create a bar chart using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(7.3, 5))
    sns.barplot(x=values, y=rnames_ed, palette="viridis")
    # ax.set(xlabel='Raster', ylabel='Value', title='Raster values from selected coordinates') 

    

    plt.title('Raster values from selected coordinates.')
    plt.ylabel('Envrionmenal Variables')
    plt.xlabel('Value')

    # Save the chart to a BytesIO object
    img = BytesIO()    
    plt.savefig(img, format='png') 
    img.seek(0)
    plt.close()

    return img

if __name__ == '__main__': 
    app.run(debug=True)
