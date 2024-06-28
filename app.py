from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('traffic.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    data1 = request.form['day']
    if(data1=="Sunday"):
        data1 = [0, 0, 0, 1, 0, 0, 0]
    elif(data1=="Monday"):
        data1 = [0, 1, 0, 0, 0, 0, 0]
    elif (data1 == "Tuesday"):
        data1 = [0, 0, 0, 0, 0, 1, 0]
    elif (data1 == "Wednesday"):
        data1 = [0, 0, 0, 0, 0, 0, 1]
    elif (data1 == "Thursday"):
        data1 = [0, 0, 0, 0, 1, 0, 0]
    elif (data1 == "Friday"):
        data1 = [1, 0, 0, 0, 0, 0, 0]
    else:
        data1 = [0, 0, 1, 0, 0, 0, 0]
    data2 = request.form['junction']
    list2 = [data2]
    data3 = request.form['month']
    list3 = [data3]
    data4 = request.form['date_no']
    list4 = [data4]
    data5= request.form['hour']
    list5 = [data5]
    final_data = data1 + list2 + list3 + list4 + list5
    arr = np.array([final_data])
    prediction = model.predict(arr)
    prediction = prediction.round()
    return render_template('prediction.html', pred= prediction)



if __name__ == '__main__':
    app.run(debug=True)