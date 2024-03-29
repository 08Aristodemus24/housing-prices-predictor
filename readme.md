# **WEB APPLICATION DEPLOYED AND IN PRODUCTION** live at url https://housing-prices-predictor.onrender.com/

# Demo:
<img align="center" src="https://github.com/08Aristodemus24/housing-prices-predictor/blob/master/web%20app%201.png" alt="michaelcueva" style="max-width: 100%"/>
<br/>
<img align="center" src="https://github.com/08Aristodemus24/housing-prices-predictor/blob/master/web%20app%202.png" alt="michaelcueva" style="max-width: 100%"/>
<br/>
<img align="center" src="https://github.com/08Aristodemus24/housing-prices-predictor/blob/master/web%20app%203.png" alt="michaelcueva" style="max-width: 100%"/>

# Web App Usage:
1. because API can both be used as an application with a working UI and as an endpoint navigating instead to url https://housing-prices-predictor.vercel.app/predict/json with the necessary data payload using postman will return a json response. But navigating to https://housing-prices-predictor.vercel.app/predict once input data is entered will redirect to the base url https://housing-prices-predictor.vercel.app/ with the predicted value
2. local machine usage will be to simply navigate to `/server-side` directory then assuming all dependencies are installed by following **Source Code Usage** instructions below just run python index.py and wait for `localhost/127.0.0.1` server at port `5000` to start e.g. `https://127.0.0.1:5000`


# Source Code Usage:
1. clone repository with `git clone https://github.com/08Aristodemus24/housing-prices-predictor.git`
2. navigate to directory with `readme.md` and `requirements.txt` file
3. run command; `conda create -n <name of env e.g. housing-prices-predictor> python=3.11.2`. Note that 3.11.2 must be the python version otherwise packages to be installed would not be compatible with a different python version
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate housing-prices-predictor`
6. check if pip is installed by running `conda list -e` and checking list
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in

# Things to implement:
- instead of a linear model implement a function to engineer new features that results in a more polynomial equation to use as our model:
- note that we have to normalize data first before passing data to map_feature() which engineers new features out of the current features
to make the equation more polynomial
- I'm actually passing unnormalized X values in the test model so I need to find a way to recover previous standard deviation and mean calculated from the training the data which was used to normalized both training and cross validation data

# References:
* https://flask.palletsprojects.com/en/3.0.x/patterns/wtforms/