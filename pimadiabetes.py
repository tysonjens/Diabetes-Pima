
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy score
get_ipython().magic('matplotlib inline')

def violin_plot_binary(categorical_var, continuous_var, df):
    # Draw a nested violinplot and split the violins for easier comparison
    image = sns.violinplot(x=categorical_var, y=continuous_var, data=df, split=True,
                   inner="quart")
    sns.despine(left=True)
    return image

def plotroc(FPR, TPR):
    roc_auc = auc(FPR, TPR)
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('img/ROCtree.png')
    
def try_lasso_hyperparam(X_train, y_train, X_test, y_test, params_to_try,):
    aucs = []
    for param in params_to_try:
        mod = LogisticRegression(penalty='l1', C=param)
        mod.fit(X_train, y_train)
        y_test_preds = mod.predict_proba(X_test)[:,1]
        aucs.append(roc_auc_score(y_test, y_test_preds))
    return aucs

def bootstrap_ci_coefficients(X_train, y_train, num_bootstraps):
    X_train = X_train.values
    y_train = y_train.values
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = LogisticRegression()
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_[0])
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates

def plot_bootstrap_estimate(col_name_list, bootstrap_coefficient_array):
    number_of_graphs = len(col_name_list)
    fig, axes = plt.subplots(int((len(col_name_list)+1)/3),3, figsize=(10,10))
    count = 0
    for m, ax in zip(col_name_list, axes.flatten()):
        ax.hist(bootstrap_coefficient_array[:,count], bins=30)
        ax.set_title(m)
        count += 1
    return fig


if __name__ = "__main__":
    
diab = pd.read_csv('data/diabetes.csv')

diab.info()

diab.describe()

diab.columns

pairplot = sns.pairplot(diab[['Outcome', "Insulin", "BMI", "Age"]][50:200])

## print corr heat map
sns.set(style="white")
# Compute the correlation matrix
corr = diab.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


glucimg = violin_plot_binary("Outcome", "Glucose", diab)

bmiimg = violin_plot_binary("Outcome", "BMI", diab)


######## Begin Models ##########

X = diab.drop('Outcome', axis=1)
y = diab['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y)


mod_2 = LogisticRegression()
mod_2.fit(X_train, y_train)
y_test_preds2 = mod_2.predict_proba(X_test)[:,1]
FPR_2, TPR_2, thresholds_2 = roc_curve(y_test, y_test_preds2)

mod_2.coef_

plotroc(FPR_2, TPR_2)


params = np.linspace(.01,2,100)
aucs = try_lasso_hyperparam(X_train, y_train, X_test, y_test, params)

## Plot best tuning param
figpt = plt.figure(figsize=(10,5))
ax1 = figpt.add_subplot(111)
ax1.plot(params, aucs)
ax1.set_title('ROC-Area under curve for various hyperparameters "C"')
ax1.set_ylabel('Area Under Curve')
ax1.set_xlabel('Tuning Parameter "C"')


bootstrapped_coefficients = bootstrap_ci_coefficients(X_train, y_train, 1000)

bscoefs = plot_bootstrap_estimate(X_train.columns, bootstrapped_coefficients)


## Model 2 - Decision Tree Method
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']

diab_tree = tree.export_graphviz(clf, out_file=None, feature_names=cols, 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(diab_tree)  
graph 



tree_preds = clf.predict_proba(X_test)[:,1]
FPR_tree, TPR_tree, thresholds_tree = roc_curve(y_test, tree_preds)
plotroc(FPR_tree, TPR_tree)

## Model 3 - Gradient Boost


grb = GradientBoostingClassifier()

grb.fit(X_train, y_train)

gbr_preds = rfor.predict_proba(X_test)[:,1]

FPR_gbr, TPR_gbr, thresholds_gbr = roc_curve(y_test, gbr_preds)

plotroc(FPR_gbr, TPR_gbr)

## Grid Search

gb_grid = {'learning_rate': [0.01, 0.1, 0.5, 1], 
                    'max_depth': [3, 2, None],
                      'max_features': ['sqrt', 'log2', None],
                      'min_samples_split': [2, 4],
                      'min_samples_leaf': [1, 2, 4],
                      'n_estimators': [10, 20, 40, 80],
                      'random_state': [1]}

gb_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                             gb_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='roc_auc')
gb_gridsearch.fit(X_train, y_train)

print ("best parameters:", gb_gridsearch.best_params_)

best_gb_model = gb_gridsearch.best_estimator_

best_gb_model.fit(X_train, y_train);

gbrbest = best_gb_model.predict_proba(X_test)[:,1]
FPR_gbrbest, TPR_gbrbest, thresholds_gbrbest = roc_curve(y_test, gbrbest)
plotroc(FPR_gbrbest, TPR_gbrbest)

np.mean(cross_val_score(best_gb_model, X_train, y_train, scoring='accuracy'))

y_preds_gbr = best_gb_model.predict(X_test)

y_preds_tree = clf.predict(X_test)

y_preds_mod2 = mod_2.predict(X_test)

accuracy_score(y_test, y_preds_gbr)

accuracy_score(y_test, y_preds_tree)

accuracy_score(y_test, y_preds_mod2)

