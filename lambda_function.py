import json
import boto3
import joblib

def predict_user_id(event, context):
    s3 = boto3.client('s3')
    model_type = event['Model']
    ht_mean = event['HT']['Mean']
    ht_std = event['HT']['STD']
    ppt_mean = event['PPT']['Mean']
    ppt_std = event['PPT']['STD']
    rpt_mean = event['RPT']['Mean']
    rpt_std = event['RPT']['STD']
    rrt_mean = event['RRT']['Mean']
    rrt_std = event['RRT']['STD']
    
    input_data = [ht_mean, ht_std, rrt_mean, rrt_std, ppt_mean, ppt_std, rpt_mean, rpt_std]
    
    if model_type == 'SVM':
        model = joblib.load(s3.get_object(Bucket='mymodelstorage', Key='svc.joblib')['Body'])
    elif model_type == 'RF':
        model = joblib.load(s3.get_object(Bucket='mymodelstorage', Key='rfc.joblib')['Body'])
    elif model_type == 'XGB':
        model = joblib.load(s3.get_object(Bucket='mymodelstorage', Key='xgb.joblib')['Body'])
    else:
        return {
            'statusCode': 400,
            'body': 'Invalid model type'
        }
    
    prediction = model.predict([input_data])[0]
    
    return {
        'statusCode': 200,
        'body': json.dumps({'UserID': int(prediction)})
    }
