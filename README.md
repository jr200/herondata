# herondata exercise 

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

