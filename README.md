# 476_573_TMDB_predictor

In this project, Python 3 is used.
First of all, Python must be downloaded and installed.

sudo apt-get install python3

Pip can be downloaded from below command:

sudo apt install python3-pip


Commands to download libraries:

pip3 install pandas
pip3 install numpy
pip3 install sklearn
pip3 install collections
pip3 install xgboost
pip3 install lightgbm
pip3 install catboost

Before run:
Take data files to run directory. 

To create TMDB model and write a file submission:
Run this command:
python3 TMDBPrediction.py

Results will be written in these files:
XGB prediction : submission_xgb.csv
LGB prediction : submission_lgb.csv
CAT prediction : submission_cat.csv
Weighted prediction according to three models above :  submission_xgb_lgb_cat.csv

To create IMDB model and write a file submission:
Run this command:
python3 IMDBPrediction.py

Results will be written in this file:
Weighted prediction according to three models(xgb, lgb, cat) :  submission_xgb_lgb_cat_imdb.csv




