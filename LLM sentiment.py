#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 13:39:52 2025

@author: mariadomardealmeidavau
"""
 
#%% First, you need to import the necessary modules and libraries (the tools) to run the code. 
#You are importing tools to allow the use of LLMs (Llama Maverick) for sentiment analysis.
#Please bear in mind that, due to the nature of the data used,you might have to use an ablated version of a model, or alter certain model parametres; otherwise, the requests sent to the model may be denied,due to the language the data contains.

import os
from openai import OpenAI

import pandas as pd

#%% Next, you give your computer information on where and how to access the LLM model you are deploying. 
#Replace your-resource-name below with the information relevant to your computer.
 
endpoint = "https:// your-resource-name.services.ai.azure.com/openai/v1/"

model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"

deployment_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"


#%% Now, you will need to provide the computer an API key, which allows requests to be put through to the LLM model. The API key is obtained after you have signed up on the respective provider's platform; it also often requires you to add credit to use the model.
#Replace AZURE_API_KEY  with your API key.

apikey = "AZURE_API_KEY" 

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=apikey)

#%% Next, you need to define the prompt to instruct the model on the sentiment analysis task. We recommend testing your prompt before deploying it to ensure it returns valid information. The code below can be used for the validation process.
#After defining the prompt, you define a function to analyse your data; a function is a set of instructions to be repeated.
#We have called the function 'sentiment', and the variable it is applied to is 'text'. Within the function, the LLM is given a role, and it is specified what it needs to do. The function returns the outcome of the sentiment analysis task as stated in the prompt in the variable called 'message'.
#Refer to model documentation for information on additional parameters that can be adjusted. 

prompt = "Classify each sentence based on its sentiment value (positive, neutral, negative) and attribute a sentiment score, based on its intensity (ranging from 1 to 5). Then extract aspect and segment pairs for each sentence. Provide a short explanation for your analysis. Structure your answer as follows: Sentiment: , Score: , Aspect: , Segment: , Explanation: . Do not include anything else."

def sentiment(text):
  response = client.chat.completions.create(
      model = deployment_name,
      messages = [ 
          {"role": "system",
          "content": "You are a sentiment analysis expert. You have been tasked with analysing sentiment in social media text."},
          {"role": "user", "content": prompt+"\n\n"+text}
          ],)

  message = response.choices[0].message.content
  return message

#%% To analyse your data, meaning you apply the sentiment function to your data, you first read your data. You will use pandas for this. Specify first the file path to your dataset and tell pandas the format of the file it has to read. Replace yourdata.csv with the path of your data set.
#Then, store the data in a data frame, here called 'df'.
#Because the file can have more than one column of data, you also need to specify which data within the file needs to be analysed. Here, we called the column with the information to be analysed text.

df = pd.read_csv("yourdata.csv") 
text = df['text']

#To store our results, we create an empty list to be used later.
lresults = []

#%% We will be using a loop (repetition of a set of instructions) to analyse each of the data points (i.e., each sentence in the data in the text column).
#The loop will run for as long as there is data to analyse. You can adapt the number of times that the loop runs in range().
#For every sentence (di) in the data (df), the previously described function 'sentiment' is applied.
#The results of the sentiment analysis, is being saved in the empty list, lresults.
#additional_params is included to ensure the computer recognises the data as a string and to improve the formatting of what is given to the function and, subsequently, the model used.
#the try and except are included to avoid the code from stopping when it runs into an exception or error.

for di in range(len(df)):
    phrase = text[di]
    additional_params= sentiment('Sentence: '+str(phrase))
    try:
        result = additional_params.split("\n\n")
    except: 
        result = {}
    try: 
        lresults.append(result[0])
    except Exception as e:
        lresults.append("Not determined")

#%% The next segment of the code assigns the results of the sentiment analysis to a new column in the data frame, SentimentAnalysis.
#This means that, when you open your dataset, there should be a new column of the results of our sentiment analysis, that is, a score between 1 – 5.
#Replace yourdata.csv with your data file name.

df = df.assign(SentimentAnalysis = lresults)
df.to_csv("yourdata.csv ", index = False)



