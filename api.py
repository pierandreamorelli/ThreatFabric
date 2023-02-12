from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    model_type = input_data['Model']
    ht_mean = input_data['HT']['Mean']
    ht_std = input_data['HT']['STD']
    rrt_mean = input_data['RRT']['Mean']
    rrt_std = input_data['RRT']['STD']
    ppt_mean = input_data['PPT']['Mean']
    ppt_std = input_data['PPT']['STD']
    rpt_mean = input_data['RPT']['Mean']
    rpt_std = input_data['RPT']['STD']
    input_array = [ht_mean, ht_std, rrt_mean, rrt_std, ppt_mean, ppt_std, rpt_mean, rpt_std]
    
    if model_type == 'SVM':
        model = joblib.load('models/svc.joblib')
    elif model_type == 'RF':
        model = joblib.load('models/rfc.joblib')
    elif model_type == 'XGB':
        model = joblib.load('models/xgb.joblib')
    else:
        return 'Invalid model type', 400
    
    prediction = model.predict([input_array])[0]
    
    return jsonify({'UserID': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
