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
