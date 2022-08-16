# 2022-08-16

```
pyenv local 3.9.7
poetry env use 3.9.7
poetry init

poetry add scipy==1.6.1 scikit-learn pandas matplotlib statsmodels jellyfish transformers
poetry add --group dev jupyterlab

poetry install
poetry shell
pip install --upgrade pip

jupyter lab
```
