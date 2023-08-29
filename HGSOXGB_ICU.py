
import numpy as np
from Metropolish_HGSO import HGSO_Metropolish_diverse_stable
from Metropolish_HGSO import initialize_population_MH  # Assuming the functions are in this module
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

import pandas as pd

# Load the CSV files into dataframes
#train_df = pd.read_csv("train.csv")
#test_df = pd.read_csv("test.csv")

df = pd.read_csv('train.csv')
df.replace(to_replace=' ', value=np.nan, inplace=True)

df1 = pd.read_csv('test.csv')
df1.replace(to_replace=' ', value=np.nan, inplace=True)

df.rename(columns={
    'INTUBADO': 'INTUBATED',
    'ASMA': 'ASTHMA',
    'HIPERTENSION': 'HYPERTENSION',
    'Age': 'AGE',
    'Sex': 'SEX'
},
          inplace=True)
df1.rename(columns={
    'INTUBADO': 'INTUBATED',
    'ASMA': 'ASTHMA',
    'HIPERTENSION': 'HYPERTENSION',
    'Age': 'AGE',
    'Sex': 'SEX'
},
           inplace=True)
X_train = df.drop(columns=['ICU'])
y_train = df['ICU']
X_test = df1.drop(columns=['ICU'])
y_test = df1['ICU']


def objective_function(params):
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])
    subsample = params[3]
    colsample_bytree = params[4]
    gamma = params[5]
    min_child_weight = int(params[6])
    
    xgb = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                        min_child_weight=min_child_weight, n_jobs=-1, eval_metric='logloss')
    
    # Get predicted labels for each sample using 5-fold cross-validation
    predictions = cross_val_predict(xgb, X_train, y_train, cv=5)
    
    # Compute the Cohen's Kappa score
    kappa_score = cohen_kappa_score(y_train, predictions)
    
    # Compute the objective function value
    kappa_loss = 1 - kappa_score
    
    return kappa_loss


# Define the hyperparameters bounds
lb = np.array([1, 0.001, 2, 0.5, 0.5, 0, 1])
ub = np.array([1000, 1, 15, 1, 1, 1, 30])

# Parameters for HGSO
N = 24
Max_iter =20
dim = 7

# Running the HGSO optimization
best_solution, best_values, Convergence_XGB = HGSO_Metropolish_diverse_stable(N, Max_iter, lb, ub, dim, objective_function)


print(" best_values: ", best_values)

print("Best Loss (Five-fold Kappa Loss):", best_solution)

n_estimators = int(best_values[0])
learning_rate = best_values[1]
max_depth = int(best_values[2])
subsample = best_values[3]
colsample_bytree = best_values[4]
gamma = best_values[5]
min_child_weight = int(best_values[6])
#alpha=best_values[7] 
optimized_xgb_model_usingHGS = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                    min_child_weight=min_child_weight, n_jobs=-1, eval_metric='logloss')

optimized_xgb_model_usingHGS.fit(X_train, y_train)

# Making predictions on the test set
y_pred = optimized_xgb_model_usingHGS.predict(X_test)

from sklearn.metrics import accuracy_score
# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy


from joblib import dump

HGSOXGB_clf=optimized_xgb_model_usingHGS

dump(HGSOXGB_clf, 'HGSOXGB_clf.joblib')   # Save the model 
