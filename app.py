# Importing essential libraries and modules

from flask import Flask, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import pickle
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import io
import base64
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Loading crop recommendation model


crop_recommendation_model_path = 'models/model2.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))



app = Flask(__name__)

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    return a, p, r, f

@ app.route('/trainML')
def trainML():
    dataset = pd.read_csv(r'C:\Users\sadha\OneDrive\Desktop\Crop_Recommendation Project\Data\crop_recommendation.csv')
    dataset.fillna(0, inplace = True)
    le = LabelEncoder()
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))#encode all str columns to numeric
    Y = dataset['label'].values.ravel()
    dataset.drop(['label'], axis = 1,inplace=True)
    dataset.fillna(0, inplace = True)            
    X = dataset.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    sc = StandardScaler()
    X = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    knn_cls = KNeighborsClassifier(n_neighbors=2)
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    a, p, r, f = calculateMetrics("KNN", predict, y_test)
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    a1, p1, r1, f1 = calculateMetrics("Random Forest Algorithm", predict, y_test)
        
    output = ''
    output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
    output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
    algorithms = ['KNN', 'Random Forest']
    output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(a)+'</td><td><font size="" color="black">'+str(p)+'</td><td><font size="" color="black">'+str(r)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
    output+='<td><font size="" color="black">'+algorithms[1]+'</td><td><font size="" color="black">'+str(a1)+'</td><td><font size="" color="black">'+str(p1)+'</td><td><font size="" color="black">'+str(r1)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
    output+= "</table></br>"
    df = pd.DataFrame([['KNN','Precision',p],['KNN','Recall',r],['KNN','F1 Score',f],['KNN','Accuracy',a],
                       ['Random Forest','Precision',p1],['Random Forest','Recall',r1],['Random Forest','F1 Score',f1],['Random Forest','Accuracy',a1],
                      ],columns=['Algorithms','Metrics','Value'])
    df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(6, 4))
    plt.title("All Algorithms Performance Graph")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()    
    return render_template('Output.html', data=output, img = img_b64)
    

@ app.route('/index')
def index():
    title = 'CropRecommendation'
    return render_template('index.html', title=title)


@ app.route('/')
def home():
    title = 'CropRecommendation'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = request.form['nitrogen']
        P = request.form['phosphorous']
        K = request.form['pottasium']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        h= request.form['humidity']
        t= request.form['temperature']
        # state = request.form.get("stt")

        #if weather_fetch(city) != None:
            #temperature, humidity = weather_fetch(city)
        print([N, P, K,t,h, ph, rainfall])
        data = np.array([[N, P, K,t,h, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

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

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
