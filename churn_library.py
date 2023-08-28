''' 
module main library
methods to read bank data and train a ML model to predict churn
author: mulhermaonocodigo
date: August, 27, 2023
'''

# import libraries
import os
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda val: 0
        if val == "Existing Customer" else 1)
    dataframe = dataframe.drop(['Attrition_Flag'],axis=1)
    return dataframe

def perform_eda(dataframe, type_plot, columns_name):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe
            type_plot : type of plot
            columns_name: name of columns to use for each plot
    output:
            none
    '''
    if type_plot == 'histogram':
        for col in columns_name:
            plt.figure(figsize=(20, 10))
            dataframe[col].hist()
            plt.savefig("./images/eda/"+col+"_hist.png",box_inches='tight')
            plt.show()
            plt.close()
    elif type_plot == 'bar':
        plt.figure(figsize=(20, 10))
        dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig("./images/eda/"+columns_name+"_bar.png",box_inches='tight')
        plt.show()
        plt.close()
    elif type_plot == 'density':
        # Show distributions of 'Total_Trans_Ct' and add a smooth
        # curve obtained using a kernel density estimate
        plt.figure(figsize=(20, 10))
        sns.displot(dataframe[columns_name], stat='density', kde=True)
        plt.savefig("./images/eda/"+columns_name+"_density.png",box_inches='tight')
        plt.show()
        plt.close()

    elif type_plot ==  'heatmap':
        plt.figure(figsize=(20, 10))
        sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig("./images/eda/heatmap.png",box_inches='tight')
        plt.show()
        plt.close()

def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_groups = dataframe.groupby(category).mean()['Churn']
        columns_lst = [category_groups.loc[val] for val in dataframe[category]]
        dataframe[category+'_Churn'] = columns_lst
    dataframe = dataframe.drop(category_lst, axis=1)
    dataframe = dataframe.dropna()

    return dataframe

def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of features to keep

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    label_data = pd.DataFrame()
    label_data['Churn'] =  dataframe['Churn']
    train_data  = pd.DataFrame()
    train_data[response] = dataframe[response]
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                        label_data,
                                                        test_size= 0.3,
                                                        random_state=42)
    x_train.to_csv("./data/X_train.csv",index=False)
    x_test.to_csv("./data/X_test.csv",index=False)
    y_train.to_csv("./data/y_train.csv",index=False)
    y_test.to_csv("./data/y_test.csv",index=False)
    return x_train, x_test, y_train, y_test

def plot_results(model1,model2,  x_test, y_test):
    '''
    produced the roc curve for two models
    input:
            model1: vector with model trained and model name
            model2: vector with model trained and model name
            X_test: data for testing
            y_test: label for testing

    output:
             None
    '''
    plt.figure(figsize=(15, 8))
    axis_plot = plt.gca()
    plot_roc_curve(model1[0], x_test, y_test, ax=axis_plot, alpha=0.8, label=model1[1])
    plot_roc_curve(model2[0], x_test, y_test, ax=axis_plot, alpha=0.8, label=model2[1])
    plt.savefig('./images/results/plot_roc_curve.png')
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    #Random forest report
    model_list =['random_forest','logistic_regression']
    label_test_predict = [y_test_preds_rf,y_test_preds_lr]
    label_train_predict = [y_train_preds_rf,y_train_preds_lr]
    i=0
    for model_name in model_list:
        plt.rc('figure', figsize=(7, 7))
        plt.text(0.01, 1.25, str(model_name+' Train'),{'fontsize': 10},fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train,label_train_predict[i])),
                 {'fontsize':10},
                 fontproperties = 'monospace')
        plt.text(0.01, 0.6, str(model_name + ' Test'),{'fontsize': 10},fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, label_test_predict[i])),
                 {'fontsize': 10},
                  fontproperties = 'monospace')
        plt.axis('off')
        plt.savefig('./images/results/'+model_name+'.png')
        plt.close()
        i=i+1

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: list of model object containing feature_importances_ and model name
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model[0].feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, model[1]+'_importance.png'), bbox_inches="tight")
    plt.close()

def explain(model,x_test, output_pth):
    '''
    plot explain features
    input:
            model: list of model object containing feature_importances_ and model name
            x_test: pandas dataframe of X test values
            output_pth: path to store the figure

    output:
             None
    '''
    plt.figure(figsize=(20, 6))
    # Create plot title
    plt.title("Feature explainer")
    explainer = shap.TreeExplainer(model[0])
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig(os.path.join(output_pth, model[1]+'_explain.png'), bbox_inches="tight")
    plt.close()

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = { 'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }
    print("before grid")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    print("after grid")
    cv_rfc.fit(x_train, y_train)
    print("after fit")
    lrc.fit(x_train, y_train)
    print("after fit lrc")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    print("after predict")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    #plot results
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

def results_report(rfc_model,lr_model,x_train,x_test,y_test):
    '''
    plot model results: images + scores
    input:
              rfc_model: random forest trained model
              lr_model: logisti regression trained model
              x_train : X training data
              x_test: X testing data
              y_test: y testing data
    output:
              None
    '''
    plot_results([rfc_model,'Random Forest'],[lr_model,'Logistic Regression'],  x_test, y_test)
    feature_importance_plot([rfc_model,'Random Forest'], x_train, './images/results/')
    explain([rfc_model,'Random Forest'],x_test, './images/results/')
