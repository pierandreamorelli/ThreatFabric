# **Part 1 - Models Building**
The Jupyter Notebook performs keystroke dynamics analysis on a given dataset. The dataset contains information about keystroke timings from multiple users and the goal is to determine the user from their keystroke timing patterns.

### **Libraries Used**
The code uses the following libraries:

- Pandas
- Numpy
- Scikit-learn
- XGBoost

### **Data Pre-processing**
The data is loaded using the Pandas library and the features are calculated using the timestamp between press and release columns. From this, mean and standard deviation values are calculated for each feature. This processed data is then stored in a Pandas DataFrame and split into feature and target variable (UserID).

### **Model Building and Evaluation**
The code then uses three different machine learning algorithms to build classifiers. These algorithms are Support Vector Machines (SVC), Random Forest Classifier (RFC), and XGBoost Classifier. The accuracy of each model is then evaluated using the test dataset.

### **Model Persistence**
The trained models are saved to disk using the joblib library for later use. The code then demonstrates how to load a saved model and use it to make predictions.

### **Results**
The code reports the accuracy of the models, with XGBoost Classifier providing the highest accuracy. The accuracy values are printed at the end of the code for each model.

# **Part 2 - Models Deployment**
This part contains the code to test the models implemented in Part 1. The API has been implemented using Flask, a Python web framework, and joblib, a library for using Python objects in a persistent way.

There are two different implementations of the API in this repository: a local version that runs on a user's machine, and a serverless version that runs on AWS Lambda.

### **Local API**
The local API is implemented using Flask and can be run on a user's machine. The API has a single endpoint at /predict that accepts a JSON payload with the input data and returns the predicted user ID.

The input data should have the following format:

```json
{
    "Model": "RF",

    "HT": {
        "Mean": 48.43,
        "STD": 23.34
        },  

    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
        },   

    "RRT": {        
        "Mean": 124.43,
        "STD": 45.34
        },   

    "RPT": {        
        "Mean": 132.56,
        "STD": 47.12
        }
}
```

Where Model is a string that specifies which machine learning model to use (either 'SVM', 'RF', or 'XGB'), and HT, PPT, RRT, and RPT are dictionaries that contain the mean and standard deviation of each feature values.

The API uses joblib to load the saved machine learning models from disk. The models should be saved in the models/ directory and have the following file names:

- svc.joblib for the SVM model
- rfc.joblib for the Random Forest model
- xgb.joblib for the XGBoost model

### **Serverless API on AWS**

The serverless API is implemented using AWS Lambda and has been deployed on AWS. The API has a single Lambda function that predicts the user ID based on the input data provided in the form of a JSON payload.

The input data should have the same format as for the local API. The Lambda function uses boto3, the AWS SDK for Python, to load the saved machine learning models from an S3 bucket. The models should be saved in an S3 bucket with the following file names:

- svc.joblib for the SVM model
- rfc.joblib for the Random Forest model
- xgb.joblib for the XGBoost model
The S3 bucket name should be specified in the code. In the provided code, the S3 bucket name is set to mymodelstorage.


### **Testing**

The local API can be tested by running the **testapi.py** code:

```python
import requests

url = 'http://127.0.0.1:5000/predict'

input_data = {        
    "Model": "RF",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
        },
    
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
        },
    
    "RRT": {        
        "Mean": 124.43,
        "STD": 45.34
        },    
    "RPT": {        
        "Mean": 132.56,
        "STD": 47.12
        }
}

response = requests.post(url, json=input_data)

if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.text)

```

The AWS API can be tested either by using the same file (you have to replace the **url** value with the **url of the AWS Gateway**) or using the **AWS Gateway Test** function, with the same request body.

If you would like to test my code you can just run the **models_building.ipynb** to generate the 3 models, and then run the **api.py** and test it running the **testapi.py** (feel free to use different values), if you want to test it on AWS just load the 3 models on an S3 bucket and upload the **lambda_function.py**, then create an API Gateway to test it.