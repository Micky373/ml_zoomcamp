from flask import Flask, request,jsonify
import pickle

model_loc = "model1.bin"
dv_loc = "dv.bin"

with open(model_loc,'rb') as model:
    model_ = pickle.load(model)
with open(dv_loc,'rb') as dv:
    dv_ = pickle.load(dv)

app = Flask('Credict risk')

@app.route('/predict',methods=['POST'])

def predict():

    customer  = request.get_json()

    x = dv_.transform([customer])

    y_pred = model_.predict_proba(x)[0,1]

    credit_risk = y_pred >= 0.5

    result = {
        'credit_risk_probability': float(y_pred),
        'credit_risk': bool(credit_risk)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0',port=9696)

