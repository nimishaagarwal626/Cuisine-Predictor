import argparse
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Reads yummly.json and convert it into Dataframe
def read_json_file(filename):
    with open(filename, 'r') as yummly:
        yummly_data = json.load(yummly)
    return pd.DataFrame(yummly_data)

# takes dataframe and convert it into list
def convertIngredientstoList(df, input_ingredients):
    list_of_ingredients = df['ingredients']
    convertedIngredientList = [' '.join(i) for i in list_of_ingredients]
    # append the input ingredients from CLI at the end of the list
    convertedIngredientList.append(input_ingredients)
    return convertedIngredientList

# vectorize the ingredient list using TfidfVectorizer
def vectorize(ingredients_list):
    tfidf = TfidfVectorizer()   
    transformed_ingredients = tfidf.fit_transform(ingredients_list)
    return transformed_ingredients

# Calculates similarity score between input and reference data using cosine similarity feature from sklearn
def calculateSimilarityScoreAndPredictClosest(N, ingredients, input_ingredients):
    similarity_score = cosine_similarity(input_ingredients, ingredients)
    # sorts the generated array
    closest_indices = np.argsort(similarity_score, axis=1)[:,::-1][:,:N][0]
    closest_score = similarity_score[0, closest_indices]
    # maps similar index of multiple containers
    closest_cuisine = list(zip(closest_indices, closest_score))
    return closest_cuisine

# Predicts the closest cuisine
def closest_cuisine(df, ingredients_list, N):
    # calls the vectorizer
    transformed_ingredients = vectorize(ingredients_list)
    cuisine = df['cuisine']
    # calls train_test_split model from sklearn for training and testing the model.
    x_train, x_test, y_train, y_test = train_test_split(transformed_ingredients[:-1], cuisine, test_size=0.3, random_state=100)
    # Calls KNeighboursClassifier to predict the closest neighbours 
    neighbourClassifier =  KNeighborsClassifier(n_neighbors=N)
    neighbourClassifier.fit(x_train, y_train)
    predicted_data = neighbourClassifier.predict(transformed_ingredients[-1])

    # find the max from teh array
    cuisine_score = np.amax(neighbourClassifier.predict_proba(transformed_ingredients[-1])[0])
    # calls calculateSimilarityScoreAndPredictClosest utility method to get the similarity score.
    closest_data = calculateSimilarityScoreAndPredictClosest(N, transformed_ingredients[:-1], transformed_ingredients[-1])
    closest_cuisines = []
    cuisine_ids = list(df['id'])
    for id, score in closest_data:
        closest_cuisines.append((cuisine_ids[id], score))
    return predicted_data, cuisine_score, closest_cuisines

# Return the json output by rounding off the scores to 2 decimal points.
def predictedCuisines(predicted_cuisine, pred_cuisine_score, closest):
    closest_cuisine_list = [{"id": row[0], "score": round(row[1], 2)} for row in closest]
    result = {"cuisine": predicted_cuisine[0], "score": round(pred_cuisine_score, 2), "closest": closest_cuisine_list}
    print(json.dumps(result, indent=3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",type=int, required=True,help="Number of closest cuisines")
    parser.add_argument("--ingredient", type=str, action='append', required=True, help="It takes list of ingredients")
    args=parser.parse_args()
    filename = "yummly.json"
    jsondf = read_json_file(filename)
    input_ingredients = " ".join(args.ingredient)
    ingredients_list = convertIngredientstoList(jsondf, input_ingredients)
    predicted_cuisine, pred_cuisine_score, closest = closest_cuisine(jsondf, ingredients_list, args.N)
    predictedCuisines(predicted_cuisine, pred_cuisine_score, closest)