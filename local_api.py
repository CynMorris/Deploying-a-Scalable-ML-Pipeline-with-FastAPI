import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
# https://www.w3schools.com/python/module_requests.asp
r = requests.get("http://127.0.0.1:8000") # Your code here

# TODO: print the status code
# https://realpython.com/python-requests/
# Print the status code of the response
print(f"Status code: {r.status_code}")
# TODO: print the welcome message
print(f"Message: {r.json()['message']}")

# Don’t need to create a JSON file manually for this. 
# When interacting with an API, the response from API server 
# is typically in JSON format, which the requests library in Python 
# can handle directly.


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
# https://realpython.com/python-requests/
r = requests.post("http://127.0.0.1:8000/data/", json=data) # Your code here

# TODO: print the status code
# # https://realpython.com/python-requests/
#print(f"Status code: {r.status_code}")
# TODO: print the result
#print(f"Result: {r.json()['result']})
#print(f"Result: r.json())
# https://www.geeksforgeeks.org/response-text-python-requests/
#print("Result:", r.json())


# Printing the status code
print(f"Status code: {r.status_code}")
# Printing the result
#print(f"Result: {r.json()['result']}")
print(f"Response data: {r.text}")