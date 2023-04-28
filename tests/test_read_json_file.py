import pandas as pd
from project2 import read_json_file

def test_read_json_file():
    df = read_json_file('docs/test_yummly.json')
    assert df.shape == (2, 3)
    assert set(df.columns) == {'id', 'cuisine', 'ingredients'}

    assert df.iloc[0]['cuisine'] == 'greek'
    assert df.iloc[0]['ingredients'] == ["romaine lettuce", "black olives", "grape tomatoes", "garlic", "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles"]
