import pandas as pd
from project2 import convertIngredientstoList

def test_build_corpus():
    df = pd.DataFrame({'ingredients': [['wheat', 'onion', 'garlic'], ['rice', 'veggies']]})
    input_ingredients = 'water'
    expected_output = ['wheat onion garlic', 'rice veggies', 'water']
    assert convertIngredientstoList(df, input_ingredients) == expected_output
