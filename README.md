rhenhw3
==============================

MIS 6900 aap course - Homework 3 - Simulation of border crossing
==============================

Model requirements can be found in:
	requirements.txt

	
The original simple border crossing model with fixed configuration can be found in:
	notebooks\rhenhw3_simulation.ipynb
	
A more complex model that allows for the input of configuration files can be found in:
	src\rhenhw3\border_crossing_model_4.py

	
Various input configuration files can be found in:
	src\rhenhw3\input\
	
	
	
Eg. To run model with base configurations from python from this directory:
		python border_crossing_model_4.py --config input/base.cfg	

	
Various output files can be found in:
	src\rhenhw3\output\
	




Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so rhenhw3 can be imported
    ├── src                <- Source code for use in this project.
    │   ├── rhenhw3 <- Main package folder
    │   │   └── __init__.py    <- Marks as package
    │   │   └── rhenhw3.py  <- Just some placeholder Python source code file
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on a simplified version of the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
