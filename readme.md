# **PRODUCTION DONE. AWAITING DEPLOYMENT**  

# Usage:
1. clone repository with `git clone https://github.com/08Aristodemus24/housing-prices-predictor.git`
2. navigate to directory with `readme.md` and `requirements.txt` file
3. run command; `conda create -n <name of env e.g. housing-prices-predictor> python=3.11.2`. Note that 3.11.2 must be the python version otherwise packages to be installed would not be compatible with a different python version
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate housing-prices-predictor`
6. check if pip is installed by running `conda list -e` and checking list
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in

# Things to implement in the future:
- instead of a linear model implement a function to engineer new features that results in a more polynomial equation to use as our model:
- note that we have to normalize data first before passing data to map_feature() which engineers new features out of the current features
to make the equation more polynomial
- add normalizer for predictor

# References:
* https://flask.palletsprojects.com/en/3.0.x/patterns/wtforms/