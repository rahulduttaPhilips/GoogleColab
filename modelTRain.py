import json
import pandas as pd
import collections
from sklearn.preprocessing import MultiLabelBinarizer


with open('recipes.json') as re:
    recipes = json.load(re)


recipesDict_ = collections.defaultdict()
for item in recipes:
    desc = str(item["description"] or '')
    for step in item["steps"]:
        desc = desc + str(step["description"] or '')
    recipesDict_[desc] = item["tags"]

recipesDf = pd.DataFrame(recipesDict_.items())
recipesDf.columns = ['desc', 'tags']
mlb = MultiLabelBinarizer(sparse_output=True)
recipesDf = recipesDf.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(recipesDf.pop('tags')), index=recipesDf.index, columns = mlb.classes_))
print(recipesDict_)