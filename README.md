### An Ensemble Based Implementation for Classification

## An Overview of the foldering structure
All scripts are executed from the root dir of the proejct.
All data versioning is done in the folder named `./dbs/`.
There we have a subdirectory named `raw/` which will house 
the data. The data should be saved as follows `./dbs/raw/db.csv`.
In the folder `axl` there will be a `txt` file that houses all the 
neccessary packages to execute the whole pipeline. In the folder
`jnb`, all jupyter notebooks will be saved. Finally in the `src`
folder all python scripts and all the models will be founds.

## Pipeline Execution
1. `python3 src/data/preprocess.py`
1. `python3 src/data/process.py`
1. `python3 src/main.py`

## The output
The output prediction will be saved as an `npy` file
which will have the last 200 predictions in order.
To load them in python, one can use the following code<br>
`import numpy as np; y_hat = np.load("./dbs/cooked/preds.npy")`
