## CS5293sp23 – Project2

## Name: Nimisha Agarwal

## Project Description:
This project take a json file yummly.json in this case obtained from https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json which contains ids, cuisine names and the ingredients used. Taking this file as a reference point, based on the ingredient input and number of closest cuisine input from command line, this project finds out the N closest cuisine using KNeighboursClassifier and also their scores using cosineSimilarity.

## How to install:
Pandas installation: pipenv install Pandas
numpy installation: pipenv run numpy
pytest installation: pipenv install pytest
scikit-learn installation: pipenv install scikit-learn

## How to run:
* To run the project: pipenv run python project2.py --N 5 --ingredient wheat --ingredient salt --ingredient rice 
* To run the pytests: pipenv run python -m pytest

# Video:

## Functions
# project2.py \
* read_json_file(filename) - This function takes the filename as input parameter which is a json file and convert it into a dataframe which is then returned.

* convertIngredientstoList(df, input_ingredients) - This function takes the dataframe returned by the above function and the input_ingredients(passed through command line) as input parameters and convert the ingredients of dataframe into a list, It also appends the input_ingredient at the end of the list for easy accessibility, then return the updated list.

* vectorize(ingredients_list) - This function is a utility function that takes ingredients_list  as input parameter and vectorize the list using TfidfVectorizer and then return the vectorized list.

* calculateSimilarityScoreAndPredictClosest(N, ingredients, input_ingredients) - This function is a utility function thattakes N(passed from command line), ingredients(converted yummly data) and input_ingredients(from the command line) as input parameter and then calculates the cosine similarity between input_ingredients and ingredients, sort it based on the similarity score using argsort, calculates the closest score and return that list. 

* closest_cuisine(df, ingredients_list, N) - This function takes dataframe, ingredients_list and N as input parameter. It then calls the vectorize method and calls train_test_split model from sklearn for training the data model. It then uses KNeighbours classifier for predicting the class of a new data point based on the classes of its nearest neighbors in the training set and return the predicted data along with the similarity score and closest cuisines.

* predictedCuisines(predicted_cuisine, pred_cuisine_score, closest) - This function takes the returned values from the above method as input parameter to generate the expected json output.

# Tests:
* test_read_json_file.py - This file tests the read_json_file(filename) function of project2.py. It takes filename in as input parameter and calls read_json_file(filename) and assert the shape and the values of dataframe.

* test_closestcuisine.py - This file tests the closest_cuisine(df, ingredients_list, N) function from project2.py. It takes df, ingredients_list, N as input and assert the expected output which is predicted_data, cuisine_score and closest_cuisines.

* test_converted_ingredient_list.py - This file tests the convertIngredientstoList(df, input_ingredients). It takes dataframe and input_ingredients as input parameter and assert the expected out which should be the list that has the data from json file appended with the input ingredients.

## Bugs And Assumptions
# Assumptions:
* Ingredients passed from command line argument should be texts(no special characters) and it should be present in yummly.json

# Bugs:
* The code may give issues if ingredients passed are special characters.