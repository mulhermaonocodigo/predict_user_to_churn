# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project open a data file with user information of credit card and train two different models to predict churn.



## Files and data description
Overview of the files and data present in the root directory. 


./                    - K8s deployment files
├── churn_script_logging_and_test.py        - File to test all methods
├── requirements_py3.8.txt        - File with all packages required for running this project
└── churn_library.py           - main file with all methods to train and test a model to predict churn


data/                   
├── bank_data.csv            - input data
├── X_train.csv              - saved data for training
├── X_test.csv               - saved data for testing
└── y_test.csv               - label for testing
└── y_train.csv              - label for training

images/   
    └── eda                         
        ├── Churn_hist.png                   - histogram Churn figure
        ├── Customer_Age_hist.png            - histogram Age figure
        ├── heatmap.png                      - heatmap figure
        ├── Marital_Status_bar.png           - marital status figure
        └── Total_Trans_Ct_density.png.png   - density figure       
    └── results                       
        ├── logistic_regression.png            - logistic regression result
        ├── random_forest.png                  - randon forest result
        ├── plot_roc_curve.png                 - roc curve of random forest and logistic regression
        ├── Random Forest_explain.png          - model explanation 
        └── Random Forest_importance.png       - feature importance

models/             
    ├── logistic_model.pkl           - trained model logistic regression
    └── rfc_model.pkl                - trained model random forest



## Running Files

#### to  set up:

python3.8 -m venv clean_code_env

source clean_code_env/bin/activate

pip install -r requirements_py3.8.txt

#### to run: 

python churn_script_logging_and_test.py





