# Industrial Data Science Exercises

This repository contains all Jupyter notebooks created to solve the exercises for the Industrail Data Science lecture at TU Dortmund (WiSe 23/24). Each exercise is orgnaised it it's folder, that contains all required data. Exercise sheets are not included (since I am not entirely sure if I am allowed to publish those).

## Setup

The repository contains Jupyter notebooks and all data. To get set up, you need to install python Jupyter kernel for the notebooks to use. A simple venv with the Jupyter package installed is recommended. Additionally commonly used libraries (like pandas, plotly and sklearn) are used. The full list of dependencies can be found in the [requirements.txt](requirements.txt).

```shell
# setup venv
python -m venv .venv

# activate venv
source .venv/bin/activate

# install jupyter (and requirements)
pip install jupyter
pip install -r requirements.txt
```

## How to Run

With Jupyter and the requirements installed the notebooks should (mostly) run fine. Rerunning the same blocks may lead to some unwanted side effect, even though most of the code mutating dataframes (or similar) should be guarded with `if` statements.
