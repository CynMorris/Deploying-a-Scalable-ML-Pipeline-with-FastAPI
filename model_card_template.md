# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistical Regression model was used   
Model was trained on 1994 Census Data (https://archive.ics.uci.edu/dataset/20/census+income)  
The goal was to predict wheteher a person makes over 50k a year based on other variables (age, education, occupation, etc)  

## Intended Use
For educational purposes: Udacity  
To learn about ML deployment in pipelines  
To predict the salary category of a person based on their features 

## Training Data
Census Data from 1994 was used  
There are 14 Features  
Data is Categorical and numerical

## Evaluation Data
The test set split was from the original csv  
It composed 20% of the original csv data  

## Metrics
Precision, recall, and F1 score    
On the test data, the model achieved a precision of 0.7376, a recall of 0.6.288, and an F1 score of 0.6789    

## Ethical Considerations 
The census data that itself may included biases based on such features as gender and occupation. The model should be used for educational purposes only unless further investigation is done.

## Caveats and Recommendations
Outside factors could contribute to imalance and bias. 
