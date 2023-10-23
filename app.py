import pandas as pd 
import numpy as np
import xgboost as xgb
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
	return 'Welcome to the best AI for predicting who the best player is'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(sepal_length, sepal_width, petal_length, petal_width): 

	prediction = classifier.predict( 
		[[sepal_length, sepal_width, petal_length, petal_width]]) 
	print(prediction) 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("FootBall Player Prediction") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:darkblue;padding:13px"> 
	<h1 style ="color:gold;text-align:center;">Streamlit Best Player ML App </h1> 
	</div> 
	"""
	
	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	sepal_length = st.number_input("Enter player passing score", min_value=0, max_value=100, value=25)
	sepal_width = st.number_input("Enter player dribbling score", min_value=0, max_value=100, value=25)
	petal_length = st.number_input("Enter player shooting score", min_value=0, max_value=100, value=25)
	petal_width = st.number_input("Enter player potential score", min_value=0, max_value=100, value=25)	
	result ="" 
	

	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		result = prediction(sepal_length, sepal_width, petal_length, petal_width) 
	st.success('The output is {}'.format(result)) 
	
if __name__=='__main__': 
	main() 
