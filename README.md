# herondata exercise 2022-08-16 20:30

- brainstorming/ideas are in `solution.ipynb`
- i've converted this to a python script `solution.py` which exposes the function: 

```
identify_recurring_transactions(transactions, ids_only=True)
```

it can be called like this, for example:

```
result = identify_recurring_transactions(read_transactions('example.json'))
print(result)
```

## Setup

```
pyenv local 3.9.7
poetry env use 3.9.7
poetry init

poetry add scipy==1.6.1 scikit-learn pandas matplotlib statsmodels jellyfish transformers
poetry add --group dev jupyterlab nbconvert

poetry install
poetry shell
pip install --upgrade pip

jupyter lab
```
## Notebook to Script

```
jupyter nbconvert --to script solution.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags exclude_from_script
```

## Problem

### Brainstorming


**Problem**

3. (bonus) Please include unit tests

4. (bonus) Please discuss the following:
    1. How would you measure the accuracy of your approach?

- need more data
    - training vs test set
    - could generate known data and insert it into real transaction sets

    2. How would you know whether solving this problem made a material impact on customers?

    3. How would you deploy your solution?
    4. What other approaches would you investigate if you had more time?