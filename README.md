# Crop-prediction

This application predicts the crop based on the soil's fertile condition.

# Model

We are using a `Random regressor model`. The score is `0.99`.

# Notes

Since the `label` column in the dataset contains string values, therefore we are converting them to numbers and assigning each crop with a number as below
```
      'rice':1,
      'maize':2,
      'jute':3,
      'cotton':4,
      'papaya':5,
      'orange':6,
      'apple':7,
      'muskmelon':8,
      'watermelon':9,
      'grapes':10,
      'mango':11,
      'banana':12,
      'pomegranate':13,
      'lentil':14,
      'blackgram':15,
      'mungbean':16,
      'mothbeans':17,
      'pigeonpeas':18,
      'kidneybeans':19,
      'chickpea':20,
      'coffee':21
````
So if the model gives out the output as `10`, then the crop will be `grapes`.
