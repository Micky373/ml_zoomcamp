from flask import Flask, request, jsonify
import pickle

f_in = 'model_C=(1.0).bin'

with open(f_in,'rb') as f_in:
    (dv,model) = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict',methods=['POST'])
def predict():

    customer = request.get_json()

    X = dv.transform([customer])

    y_pred = model.predict_proba(X)[0,1]

    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    } 

    return jsonify(result)

# This line enables the app.run to be excuted when the predict.py file is called

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)