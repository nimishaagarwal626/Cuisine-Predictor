import pandas as pd
from project2 import closest_cuisine

def test_closest_cuisine():
    df = pd.DataFrame({'id': [0, 100, 200], 'cuisine': ['italian', 'japanese', 'indian'], 
                       'ingredients': [['tomato', 'onion', 'garlic'], ['fish', 'soy sauce'], ['curry', 'rice']]})
    corpus = ['tomato onion garlic', 'fish soy sauce', 'curry rice', 'rice']
    N = 2
    expected_output = (['indian'], 0.5, [(200, 0.6191302964899972), (100, 0.0)])
    assert closest_cuisine(df, corpus, N) == expected_output
