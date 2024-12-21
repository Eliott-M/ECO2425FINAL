#!/usr/bin/env python
# coding: utf-8


# REFERENCE

# The OLS regression equations and the 2SLS regression equations were created by Angrist, J. D., Battistin, E., \& Vuri, D. (2017) in STATA
# I replicated their regressions in Python

# The regression and instrumental variables are created by Angrist, J. D., Battistin, E., \& Vuri, D. (2017). In a small moment: Class size and moral hazard in the Italian mezzogiorno. American Economic Journal: Applied Economics, 9(4), 216–249. https://doi.org/10.1257/app.20160267

# Data is from the replication package from Angrist, J. D., Battistin, E., \& Vuri, D. (2017b, October 1). Replication data for: In a small moment: Class size and moral hazard in the Italian mezzogiorno. openICPSR. https://www.openicpsr.org/openicpsr/project/113698/version/V1/view

# The regression section of code was assisted by the week One notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week One Notebooks.

# Cross validation was used in this project and the code was assisted by the week Two notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Two Notebooks.

# The ridge and lasso section of code was assisted by the week Three notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Three Notebooks.

# The regression trees and ensemble methods section of code was assisted by the week Four notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Four Notebooks.

# The DAG section of code was assisted by the week Five notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Five Notebooks.

# The refutation test subsection of the DAG section of code was assisted by the report-week6.pdf by Prof. Khazra
# Khazra, N., 2024. report-week6.pdf, Lecture Material,.


# The PSM and IPW models of code was assisted by the report-week8.pdf, report-week9.pdf and report-week10.pdf by Prof. Khazra
# Khazra, N., 2024. report-week8.pdf, Lecture Material,.
# Khazra, N., 2024. report-week9.pdf, Lecture Material,.
# Khazra, N., 2024. report-week10.pdf, Lecture Material,.
# PSM AND IPW models of code also assisted by week Seven, Eight, Nine and Ten notebooks by Prof. Khazra
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Seven Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Nine Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Ten Notebooks.

# The metalearner models (S-learner, T-learner, X-leaner and dobuly robust) of code was assisted by the report-week8.pdf, report-week9.pdf and report-week10.pdf by Prof. Khazra
# Khazra, N., 2024. report-week8.pdf, Lecture Material,.
# Khazra, N., 2024. report-week9.pdf, Lecture Material,.
# Khazra, N., 2024. report-week10.pdf, Lecture Material,.
# The metalearner models (S-learner, T-learner, X-leaner and dobuly robust) of code also assisted by week Seven, Eight, Nine and Ten notebooks by Prof. Khazra
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Seven Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Nine Notebooks.
# # Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Ten Notebooks.

# The double machine learning models (DML and simulating treatment variable) of code was assisted ssisted by the report-week11.pdf by Prof. Khazra
# # Khazra, N., 2024. report-week11.pdf, Lecture Material,.
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eleven Notebooks.

# The generalized random forests of code was assisted ssisted by the report-week12.pdf by Prof. Khazra
# # Khazra, N., 2024. report-week12.pdf, Lecture Material,.
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Twelve Notebooks.


# ChatGPT was used throughout the code file for help with code
# OpenAI. (2022, November 30). Introducing ChatGPT. ChatGPT. https://openai.com/index/chatgpt/

# Each section is marked below where each part of Prof. Khazra, notebooks were used.



# The OLS regression equations and the 2SLS regression equations were created by Angrist, J. D., Battistin, E., \& Vuri, D. (2017) in STATA
# I replicated Angrist, J. D., Battistin, E., \& Vuri, D. (2017) STATA regressions in Python

# The regression section of code was assisted by the week one notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week One Notebooks.






# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ISLP
from matplotlib.pyplot import subplots, tight_layout
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)


data=pd.read_stata('smallmo.dta')


# In[ ]:


data.head
data.info
data.describe


# In[ ]:


#look at the columns
data.columns
#see if any NaN
data.isna().sum()


# In[ ]:


#remove the NaN
data.replace([np.inf, not np.inf], np.nan, inplace=True)
data=data.dropna()
#check again to see if no more NaN
data.isna().sum()


# In[ ]:


#summary of data

print(data.describe())

# In[ ]:


#regression where response is math 

#y_math=data['answers_math_std']
#x_math=data.columns.drop('answers_math_std')
#X_math=MS(x_math).fit_transform(data)
#model_math=sm.OLS(y_math, X_math.astype(float))
#results_math=model_math.fit()
#summarize(results_math)


# In[ ]:

# new dataset with out math and italian results
#X_no_score = data.columns.drop(['answers_math_std', 'answers_ital_std', 'schoolid', 'classid','plessoid'])
# Define control variables
#X = MS(X_no_score).fit_transform(data)
# Define dependent variables
#Y = data['answers_math_std']
#model = sm.OLS(Y, X)
#results = model.fit()
#summarize(results)

# In[ ]:


#for trees
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
import sklearn.model_selection as skm
#end trees

#start DAG
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import graphviz
#end DAG

#THIS IS WHERE REPLICATION STARTS

data.loc[:, 'survey']=data['survey'].astype('category')
data.loc[:, 'grade']=data['grade'].astype('category')
data.loc[:, 'region']=data['region'].astype('category')
data.loc[:, 'segment']=data['segment'].astype('category')
data.loc[:, 'enrol_ins_snv']=data['enrol_ins_snv'].astype('category')

#create interactions
data['students:segment'] = data['students'] * data['segment']
data['students2:segment'] = data['students2'] * data['segment']

#replicate STATA variables the researchers made
Y = ['answers_math_std', 'answers_ital_std']

X = 'female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + dad_highedu + mom_unemp + mom_housew + mom_employed + m_dad_edu + m_mom_edu + m_mom_occ'

POLY = 'students + students2 + students:segment + students2:segment + segment'

#I also want to do C(enrol_ins_snv) * C(region) but the code runs forever region has 54 and enrol_ins_snv has 314 this many categories is too much but the results are close to the papers and same direction
FIXED = 'C(survey) + C(grade) + enrol_ins_snv * region'


#MY ADDITION
data['edu_gap'] = abs(data['m_mom_edu'] - data['m_dad_edu'])
data['interaction_effect_eg_clsize'] = data['edu_gap'] * data['clsize_snv']
MYADDITION = 'edu_gap'



CONTROLS = f'{X} + {POLY} + {FIXED} + {MYADDITION}'

#STATA divided class size by 10 after
data['clsize_snv'] = data['clsize_snv']/10



###
#CHAT GPT HELPED FOR THIS CLUSTERED OLS REGRESSION MAINLY LINE cluster_model_math
#PIAZZA recommended to do the smf.ols which helped a lot
model_math = smf.ols(f'answers_math_std ~ clsize_snv + interaction_effect_eg_clsize + {CONTROLS}', data=data).fit()

#This below line CHATGPT helped with
cluster_model_math = model_math.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with

answers_math_std_cluster_model = cluster_model_math.summary()

model_ital = smf.ols(f'answers_ital_std ~ clsize_snv + {CONTROLS}', data=data).fit()

#This below line CHATGPT helped with
cluster_model_ital = model_ital.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with

answers_ital_std_cluster_model = cluster_model_ital.summary()
###

print(answers_math_std_cluster_model)
print(answers_ital_std_cluster_model)

#NOW IV PART WITH 2SLS

#math and ital use same IV
#predict clsize_snv with Maimondides
first_stage = smf.ols(f'clsize_snv ~ clsize_hat + o_math + {CONTROLS}', data=data).fit()
print(first_stage.summary())
#fit values
data['clsize_snv_hat'] = first_stage.fittedvalues



data['interaction_effect_eg_clsize_hat'] = data['edu_gap'] * data['clsize_snv_hat']

#math with no interaction
second_stage_math_no_int = smf.ols(f'answers_math_std ~ clsize_snv_hat + {CONTROLS}', data=data).fit()

#This below line CHATGPT helped with
cluster_model_math_2SLS_no_int = second_stage_math_no_int.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with

print(cluster_model_math_2SLS_no_int.summary())


#now math_test_std
#stage two is made using fitted values from clsize_snv_hat
second_stage_math = smf.ols(f'answers_math_std ~ clsize_snv_hat + interaction_effect_eg_clsize_hat + {CONTROLS}', data=data).fit()

#This below line CHATGPT helped with
cluster_model_math_2SLS = second_stage_math.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with

print(cluster_model_math_2SLS.summary())

#now ital_test_std
second_stage_ital= smf.ols(f'answers_ital_std ~ clsize_snv_hat + interaction_effect_eg_clsize_hat + {CONTROLS}', data=data).fit()

#This below line CHATGPT helped with
cluster_model_ital_2SLS = second_stage_ital.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with

print(cluster_model_ital_2SLS.summary())

# WEEK 8 FIRST HALF START

# The metalearners part of code was assisted by the week 8 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.


from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression


Treat = ((data['clsize_snv'] * 10) <= 22).astype(int)
control_vars = f'{X} + {POLY} + C(grade) + enrol_ins_snv + clsize_hat + o_math + interaction_effect_eg_clsize + edu_gap + C(survey) + region'
control_vars_list = control_vars.split(' + ')

#chatgpt helped create line below I need to remove C() to get actual column name (same line from above)
control_vars_list = [apple if 'C(' not in apple else apple.split('C(')[1].split(')')[0] for apple in control_vars_list]
#chatgpt helped create line below I need to remove C() to get actual column name (same line from above)

y_math=data['answers_math_std']
x_math=data.columns.drop('answers_math_std')
x_math = data[control_vars_list]

from sklearn.linear_model import LogisticRegression, LinearRegression
T = Treat.astype(int)
Y = data['answers_math_std']
#X = x_math.columns.drop([T, Y])
X = pd.get_dummies(x_math, drop_first=True)

ps_model = LogisticRegression(C=1e6).fit(X, T)

data['propensity_score'] = ps_model.predict_proba(X)[:, 1]

data['Treatment'] = T
print(data[["Treatment", "answers_math_std", "propensity_score"]].head())

weight_t = 1/data.query("Treatment==1")["propensity_score"]
weight_nt = 1/(1-data.query("Treatment==0")["propensity_score"])
print("Original Sample Size", data.shape[0])
print("Treated Population Sample Size", sum(weight_t))
print("Untreated Population Sample Size", sum(weight_nt))

import seaborn as sns

plt.figure(figsize=(12, 8))
sns.distplot(data.query("Treatment==0")["propensity_score"], kde=False, label="Non Treated")
sns.distplot(data.query("Treatment==1")["propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check")
plt.legend();
plt.savefig("Positivity_Check")

weight = ((data["Treatment"]-data["propensity_score"]) /
          (data["propensity_score"]*(1-data["propensity_score"])))

y1 = sum(data.query("Treatment==1")["answers_math_std"]*weight_t) / len(data)
y0 = sum(data.query("Treatment==0")["answers_math_std"]*weight_nt) / len(data)

ate = np.mean(weight * data["answers_math_std"])

print("Y1:", y1)
print("Y0:", y0)
print("ATE", ate)


#bootstrap

from joblib import Parallel, delayed  # for parallel processing
data['Treatment'] = Treat.astype(int)

# define function that computes the IPTW estimator
def run_ps(df, X, T, y):
    df_encoded = pd.get_dummies(df[X], drop_first=True)
    # estimate the propensity score
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df_encoded, df[T]).predict_proba(df_encoded)[:, 1]

    weight = (df[T] - ps) / (ps * (1 - ps))  # define the weights
    return np.mean(weight * df[y])  # compute the ATE

np.random.seed(88)
# run 100 bootstrap samples
bootstrap_sample = 100
ates = Parallel(n_jobs=4)(delayed(run_ps)(data.sample(frac=1, replace=True), control_vars_list, 'Treatment', 'answers_math_std')
    for _ in range(bootstrap_sample))
ates = np.array(ates)

print(f"ATE: {ates.mean()}")
print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")

plt.clf()
sns.distplot(ates, kde=False)
plt.vlines(np.percentile(ates, 2.5), 0, 10, linestyles="dotted")
plt.vlines(np.percentile(ates, 97.5), 0, 10, linestyles="dotted", label="95% CI")
plt.title("ATE Bootstrap Distribution")
plt.legend();
plt.savefig("Positivity_Check_BOOT")


# WEEK 8 FIRST HALF END












# START WEEK 9

# The Doubly Robust Estimation section of code was assisted by the week Nine notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Nine Notebooks.

print('week 9')

from econml.iv.dr import DRIV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

Y = data['answers_math_std']
T = data['Treatment']
Z = data['clsize_hat']

pca = PCA(n_components=0.1)#my computer cant do all data it was going for over a day
X_red = pca.fit_transform(X)#now reduced size

print("driv")
#DRIV
driv = DRIV(
    model_y_xw=LinearRegression(),
    model_t_xw=LinearRegression(),
    model_z_xw=LinearRegression(),
    flexible_model_effect=LinearRegression(),
    cv=5)
print("DRIV before fit")
#driv.fit(Y=Y, T=T, Z=Z, X=X, Y_hat=Y_hat, T_hat=T_hat, Z_hat=Z_hat)
driv.fit(Y=Y, T=T, Z=Z, X=X_red)
print("drive after fit")
#estimate treatment effect
treatment_effect_DRIV = driv.effect(X_red)
print(treatment_effect_DRIV)

#ate
DRIV_ATE = treatment_effect_DRIV.mean()
print("DRIV_ATE", DRIV_ATE)



print("DRIV with LASSO as valiation test")
driv_LASSO = DRIV(
    model_y_xw=Lasso(alpha=0.1),
    model_t_xw=Lasso(alpha=0.1),
    model_z_xw=Lasso(alpha=0.1),
    flexible_model_effect=Lasso(alpha=0.1),
    cv=5)
print("DRIV before fit")
#driv.fit(Y=Y, T=T, Z=Z, X=X, Y_hat=Y_hat, T_hat=T_hat, Z_hat=Z_hat)
driv_LASSO.fit(Y=Y, T=T, Z=Z, X=X_red)
print("drive after fit")
#estimate treatment effect
treatment_effect_DRIV_LASSO = driv_LASSO.effect(X_red)
print(treatment_effect_DRIV_LASSO)
#ate
DRIV_ATE_LASSO = treatment_effect_DRIV_LASSO.mean()
print("DRIV_ATE_LASSO", DRIV_ATE_LASSO)



print("DRIV with RIDGE as valiation test")
driv_Ridge = DRIV(
    model_y_xw=Ridge(alpha=1),
    model_t_xw=Ridge(alpha=1),
    model_z_xw=Ridge(alpha=1),
    flexible_model_effect=Ridge(alpha=1),
    cv=5)
print("DRIV before fit")
#driv.fit(Y=Y, T=T, Z=Z, X=X, Y_hat=Y_hat, T_hat=T_hat, Z_hat=Z_hat)
driv_Ridge.fit(Y=Y, T=T, Z=Z, X=X_red)
print("drive after fit")
#estimate treatment effect
treatment_effect_DRIV_Ridge = driv_Ridge.effect(X_red)
print(treatment_effect_DRIV_Ridge)
#ate
DRIV_ATE_Ridge = treatment_effect_DRIV_Ridge.mean()
print("DRIV_ATE_Ridge", DRIV_ATE_Ridge)




print("DRIV with Random Forests as valiation test")
driv_RF = DRIV(
    model_y_xw=RandomForestRegressor(n_estimators=100, max_depth=5),
    model_t_xw=RandomForestRegressor(n_estimators=100, max_depth=5),
    model_z_xw=RandomForestRegressor(n_estimators=100, max_depth=5),
    flexible_model_effect=RandomForestRegressor(n_estimators=100, max_depth=5),
    cv=5)
print("DRIV before fit")
#driv.fit(Y=Y, T=T, Z=Z, X=X, Y_hat=Y_hat, T_hat=T_hat, Z_hat=Z_hat)
driv_RF.fit(Y=Y, T=T, Z=Z, X=X_red)
print("drive after fit")
#estimate treatment effect
treatment_effect_DRIV_RF = driv_RF.effect(X_red)
print(treatment_effect_DRIV_RF)
#ate
DRIV_ATE_RF = treatment_effect_DRIV_RF.mean()
print("DRIV_ATE_Random_Forest", DRIV_ATE_RF)

# END WEEK 9


# START PSM Propensity Score Matching

# The PSM part of code was assisted by the week 8 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.

print("PSM")
CMDATA=data.sample(n=10000, random_state=88)
from causalinference import CausalModel
cm = CausalModel(
    Y=CMDATA["answers_math_std"].values,
    D=CMDATA["Treatment"].values,
    X=CMDATA[["propensity_score"]].values
)
cm.est_via_matching(matches=1, bias_adj=True)
print(cm.estimates)


# END Propensity SCORE MATCHING PSM






# START WEEK 11 DML

# The DML part of code was assisted by the week 11 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eleven Notebooks.

print("REPORT 11")

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# The 2 functions below part of code was assisted by the week 8 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.

def cumulative_gain(data, prediction, y, t, min_periods=30, steps=100):
    size = data.shape[0]
    ordered_data = data.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_data.head(rows), y, t) * (rows/size) for rows in n_rows])

def elast(data, y, t):
    return (np.sum((data[t] - data[t].mean()) * (data[y] - data[y].mean())) /
            np.sum((data[t] - data[t].mean()) ** 2))



y = "answers_math_std"
T = "clsize_snv"
X = ['female', 'm_female', 'immigrants_broad', 'm_origin', 'dad_lowedu', 'dad_midedu', 'dad_highedu',
     'mom_unemp', 'mom_housew', 'mom_employed', 'm_dad_edu', 'm_mom_edu', 'm_mom_occ', 'students',
     'students2', 'edu_gap', 'interaction_effect_eg_clsize']


categorical_vars = ['segment', 'survey', 'grade', 'enrol_ins_snv', 'region']

train, test = train_test_split(data, test_size=0.1, random_state=88)

# LINEAR DML

debias_m = LGBMRegressor(max_depth=3)

train_pred = train.assign(clsize_snv =  train[T] -
                          cross_val_predict(debias_m, train[X], train[T], cv=5)
                          + train[T].mean()) # add mu_t for visualization.

denoise_m = LGBMRegressor(max_depth=3)

train_pred = train_pred.assign(answers_math_std =  train[y] -
                               cross_val_predict(denoise_m, train[X], train[y], cv=5)
                               + train[y].mean())



final_model_cate = smf.ols(f'answers_math_std ~ clsize_snv + interaction_effect_eg_clsize + {CONTROLS}', data=data).fit()
print(final_model_cate)

cate_test = test.assign(cate=final_model_cate.predict(test.assign(clsize_snv_res=1))
                        - final_model_cate.predict(test.assign(clsize_snv_res=0)))

print(cate_test)
average_cate = cate_test['cate'].mean()
print(average_cate)

gain_curve_test = cumulative_gain(cate_test, "cate", y=y, t=T)
plt.plot(gain_curve_test, color="C0", label="Test")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("R-Learner");
plt.savefig("linear_gain_curve.png")
plt.clf()


# NON LINEAR

debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)

train_pred = train.assign(clsize_snv_res =  train[T] - cross_val_predict(debias_m, train[X], train[T], cv=5),
                          answers_math_std_res =  train[y] - cross_val_predict(denoise_m, train[X], train[y], cv=5))

model_final = LGBMRegressor(max_depth=3)

# create the weights
w = train_pred["clsize_snv"] ** 2

# create the transformed target
y_star = (train_pred["answers_math_std"] / train_pred["clsize_snv"])

# use a weighted regression ML model to predict the target with the weights.
model_final.fit(X=train[X], y=y_star, sample_weight=w);


cate_test_non_param = test.assign(cate=model_final.predict(test[X]))
print(cate_test_non_param)
average_cate_non_param = cate_test_non_param['cate'].mean()
print(average_cate_non_param)

plt.figure()
gain_curve_test_non_param = cumulative_gain(cate_test_non_param, "cate", y=y, t=T)
plt.plot(gain_curve_test_non_param, color="C0", label="Non-Parametric")
plt.plot(gain_curve_test, color="C1", label="Parametric")
plt.plot([0, 100], [0, elast(test, y, T)], linestyle="--", color="black", label="Baseline")
plt.legend();
plt.title("R-Learner");
plt.savefig("non-linear_gain_curve.png")





# The Simulated Treatment part of code was assisted by the week 11 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eleven Notebooks.

print("SIMULATED TREATEMENT")

from sklearn.model_selection import KFold
import seaborn as sns

def cv_estimate(train_data, n_splits, model, model_params, X, y):
    cv = KFold(n_splits=n_splits)
    models = []
    cv_pred = pd.Series(np.nan, index=train_data.index)
    for train, test in cv.split(train_data):
        m = model(**model_params)
        m.fit(train_data[X].iloc[train], train_data[y].iloc[train])
        cv_pred.iloc[test] = m.predict(train_data[X].iloc[test])
        models += [m]

    return cv_pred, models

debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)

y_hat, models_y = cv_estimate(train, 5, LGBMRegressor, dict(max_depth=3), X, y)
t_hat, models_t = cv_estimate(train, 5, LGBMRegressor, dict(max_depth=3), X, T)

y_res = train[y] - y_hat
t_res = train[T] - t_hat

monotone_constraints = [-1 if col == T else 0 for col in X + [T]]

model_final = LGBMRegressor(max_depth=3, monotone_constraints=monotone_constraints)
model_final = model_final.fit(X=train[X].assign(**{T: t_res}), y=y_res)


pred_test = (test
             .rename(columns={"clsize_snv":"factual_clsize_snv"})
             .assign(jk = 1)
             .reset_index() # create day ID
             .merge(pd.DataFrame(dict(jk=1, clsize_snv=np.linspace(11, 30, 15))), on="jk")
             .drop(columns=["jk"]))


pred_test.query("index==91025")
print(pred_test.query("index==91025"))



def ensamble_pred(data, models, X):
    return np.mean([m.predict(data[X]) for m in models], axis=0)

t_res_test = pred_test[T] - ensamble_pred(pred_test, models_t, X)

pred_test[f"{y}_pred"] = model_final.predict(X=pred_test[X].assign(**{T: t_res_test}))

pred_test.query("index==91025")
print(pred_test.query("index==91025"))


y_hat_test = ensamble_pred(pred_test, models_y, X)
pred_test[f"{y}_pred"] = (y_hat_test +
                          model_final.predict(X=pred_test[X].assign(**{T: t_res_test})))

pred_test.query("index==91025")
print(pred_test.query("index==91025"))

plt.figure()
np.random.seed(1)
sample_ids = np.random.choice(pred_test["index"].unique(), 10)



sns.lineplot(data=(pred_test
                   .query("index in @sample_ids")
                   .assign(max_answers_math_std = lambda d: d.groupby("index")[["answers_math_std_pred"]].transform("max"))
                   .assign(answers_math_std_pred = lambda d: d["answers_math_std_pred"] - d["max_answers_math_std"] + d["answers_math_std_pred"].mean())),
             x="clsize_snv", y="answers_math_std_pred", hue="index");
plt.savefig("estimated_treatment.png")

print(pred_test["index"].unique())


# END WEEK 11 DML

print("DML STUFF AGAIN")
print(final_model_cate)
print(average_cate)
print(average_cate_non_param)

# WEEK 12 Generalized RANDOM FORESTS START

# WEEK 12

# The GRF part of code was assisted by the week 12 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Twelve Notebooks.

print("WEEK 12")

from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

n_treatments = 2
Z = data["clsize_hat"]
X12 = data[['female', 'm_female', 'immigrants_broad', 'm_origin', 'dad_lowedu', 'dad_midedu', 'dad_highedu',
     'mom_unemp', 'mom_housew', 'mom_employed', 'm_dad_edu', 'm_mom_edu', 'm_mom_occ', 'students',
     'students2', 'edu_gap', 'interaction_effect_eg_clsize']].values
y12 = data["answers_math_std"].values
T12 = data["clsize_snv"].values

est = CausalIVForest(criterion='mse', n_estimators=400, min_samples_leaf=40,
                     min_var_fraction_leaf=0.1, min_var_leaf_on_val=True,
                     min_impurity_decrease = 0.001, max_samples=.45, max_depth=None,
                     warm_start=False, inference=True, subforest_size=4,
                     honest=True, verbose=0, n_jobs=-1, random_state=123)

est.fit(X12, T12, y12, Z=Z)

point, lb, ub = est.predict(X12, interval=True, alpha=0.01)

plt.figure()
plt.hist(point[:, 0], bins=30, color='red', edgecolor='black')
plt.savefig("hte_hist.png")
plt.show()



plt.figure(figsize=(15,5))
plt.plot(est.feature_importances(max_depth=4, depth_decay_exponent=2.0))
plt.savefig("hte_feature_importance.png")
plt.show()


import shap
explainer = shap.Explainer(est, shap.maskers.Independent(X12, max_samples=100))
shap_values = explainer(X12[:200], check_additivity=False)
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("shap_plot.png")




# WEEK 12 GENERALIZED RANDOM FORESTS END GRF








# META LEARNERS WEEK 8

# The metalearners part of code was assisted by the week 8 notebooks by Prof. Khazra
# Khazra, N., 2024. Implementation Notebooks, Lecture Material, Week Eight Notebooks.

# WEEK 8 START
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#get_ipython().system('pip install ISLP -q')
import ISLP

from matplotlib.pyplot import subplots, tight_layout

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                        summarize,
                        poly)

data=pd.read_stata('smallmo.dta')


# In[ ]:


data.head
data.info
data.describe


# In[ ]:


#look at the columns
data.columns
#see if any NaN
data.isna().sum()


# In[ ]:


#remove the NaN
data.replace([np.inf, not np.inf], np.nan, inplace=True)
data=data.dropna()
#check again to see if no more NaN
data.isna().sum()
#for trees
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
import sklearn.model_selection as skm
#end trees

#start DAG
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import graphviz
#end DAG



data.loc[:, 'survey'] = data['survey'].astype('category')
data.loc[:, 'grade'] = data['grade'].astype('category')
data.loc[:, 'region'] = data['region'].astype('category')
data.loc[:, 'segment'] = data['segment'].astype('category')
data.loc[:, 'enrol_ins_snv'] = data['enrol_ins_snv'].astype('category')
# create interactions
data['students:segment'] = data['students'] * data['segment']
data['students2:segment'] = data['students2'] * data['segment']
# replicate STATA variables the researchers made
Y = ['answers_math_std', 'answers_ital_std']
X = 'female + m_female + immigrants_broad + m_origin + dad_lowedu + dad_midedu + dad_highedu + mom_unemp + mom_housew + mom_employed + m_dad_edu + m_mom_edu + m_mom_occ'
POLY = 'students + students2 + students:segment + students2:segment + segment'
# I also want to do C(enrol_ins_snv) * C(region) but the code runs forever region has 54 and enrol_ins_snv has 314 this many categories is too much but the results are close to the papers and same direction
FIXED = 'C(survey) + C(grade) + enrol_ins_snv * region'
# MY ADDITION
data['edu_gap'] = abs(data['m_mom_edu'] - data['m_dad_edu'])
data['interaction_effect_eg_clsize'] = data['edu_gap'] * data['clsize_snv']
MYADDITION = 'edu_gap'
# string formatting it together like they did in ISLP textbook
CONTROLS = f'{X} + {POLY} + {FIXED} + {MYADDITION}'
# STATA divided class size by 10 after
data['clsize_snv'] = data['clsize_snv'] / 10
model_math = smf.ols(f'answers_math_std ~ clsize_snv + interaction_effect_eg_clsize + {CONTROLS}', data=data).fit()


#This below line CHATGPT helped with
cluster_model_math = model_math.get_robustcov_results(cov_type='cluster', groups=data['clu'])
#This above line CHATGPT helped with


answers_math_std_cluster_model = cluster_model_math.summary()
data['clsize_snv_hat'] = first_stage.fittedvalues
data['interaction_effect_eg_clsize_hat'] = data['edu_gap'] * data['clsize_snv_hat']

data['clsize_snv'] = ((data['clsize_snv'] * 10) <= 22).astype(int)  # new one

DAG_control_vars = f'{X} + {POLY} + C(survey) + C(grade) + enrol_ins_snv + region + interaction_effect_eg_clsize + edu_gap'
DAG_control_vars_list = DAG_control_vars.split(' + ')

#chatgpt helped create line below I need to remove C() to get actual column name (same line from above)
DAG_control_vars_list = [apple if 'C(' not in apple else apple.split('C(')[1].split(')')[0] for apple in DAG_control_vars_list]
#chatgpt helped create line below I need to remove C() to get actual column name (same line from above)

DAG_y_math = data['answers_math_std']
DAG_x_math = data.columns.drop('answers_math_std')
treatment = 'clsize_snv'
instruments = ['clsize_hat', 'o_math']
outcome = 'answers_math_std'
stuff = [treatment] + instruments + [outcome]
DAG_x_math = data[DAG_control_vars_list + stuff]
X_math = MS(DAG_x_math).fit_transform(data)

# NEW DAG
import dowhy
from dowhy import CausalModel

gml_graph = """
graph [
    directed 1

    node [
        id "female" 
        label "female"
    ]    
    node [
        id "m_female"
        label "m_female"
    ]
    node [
        id "immigrants_broad"
        label "immigrants_broad"
    ]
    node [
        id "m_origin"
        label "m_origin"
    ]
    node [
        id "dad_lowedu"
        label "dad_lowedu"
    ]
    node [
        id "dad_midedu"
        label "dad_midedu"
    ]
    node [
        id "dad_highedu"
        label "dad_highedu"
    ]
    node [
        id "mom_unemp"
        label "mom_unemp"
    ]
    node [
        id "mom_housew"
        label "mom_housew"
    ]
    node [
        id "mom_employed"
        label "mom_employed"
    ]
    node [
        id "m_dad_edu"
        label "m_dad_edu"
    ]
    node [
        id "m_mom_edu"
        label "m_mom_edu"
    ]
    node [
        id "m_mom_occ"
        label "m_mom_occ"
    ]
    node [
        id "students"
        label "students"
    ]
    node [
        id "students2"
        label "students2"
    ]
    node [
        id "students:segment"
        label "students:segment"
    ]
    node [
        id "students2:segment"
        label "students2:segment"
    ]
    node [
        id "segment"
        label "segment"
    ]
    node [
        id "survey"
        label "survey"
    ]
    node [
        id "grade"
        label "grade"
    ]
    node [
        id "enrol_ins_snv"
        label "enrol_ins_snv"
    ]
    node [
        id "region"
        label "region"
    ]
    node [
        id "clsize_hat"
        label "clsize_hat"
    ]
    node [
        id "clsize_snv"
        label "clsize_snv"
    ]
    node [
        id "answers_math_std"
        label "answers_math_std"
    ]
    node [
        id "o_math"
        label "o_math"
    ]
    node [
        id "interaction_effect_eg_clsize"
        label "interaction_effect_eg_clsize"
    ]
    node [
        id "edu_gap" 
        label "edu_gap"
    ]



    edge [
        source "female"
        target "clsize_snv"
    ]
    edge [
        source "female"
        target "answers_math_std"
    ]
    edge [
        source "m_female"
        target "clsize_snv"
    ]
    edge [
        source "m_female"
        target "answers_math_std"
    ]
    edge [
        source "female"
        target "m_female"
    ]
    edge [
        source "clsize_hat"
        target "clsize_snv"
    ]
    edge [
        source "grade"
        target "clsize_snv"
    ]
    edge [
        source "grade"
        target "answers_math_std"
    ]
    edge [
        source "region"
        target "answers_math_std"
    ]
    edge [
        source "region"
        target "clsize_snv"
    ]
    edge [
        source "enrol_ins_snv"
        target "clsize_snv"
    ]
    edge [
        source "enrol_ins_snv"
        target "answers_math_std"
    ]
    edge [
        source "survey"
        target "answers_math_std"
    ]
    edge [
        source "survey"
        target "clsize_snv"
    ]
    edge [
        source "clsize_snv"
        target "answers_math_std"
    ]
    edge [
        source "m_origin"
        target "immigrants_broad"
    ]
    edge [
        source "dad_lowedu"
        target "m_dad_edu"
    ]
    edge [
        source "dad_midedu"
        target "m_dad_edu"
    ]
    edge [
        source "dad_highedu"
        target "m_dad_edu"
    ]
    edge [
        source "m_dad_edu"
        target "answers_math_std"
    ]
    edge [
        source "segment"
        target "clsize_snv"
    ]
    edge [
        source "segment"
        target "answers_math_std"
    ]
    edge [
        source "students"
        target "students2"
    ]
    edge [
        source "students"
        target "clsize_snv"
    ]
    edge [
        source "students"
        target "answers_math_std"
    ]
    edge [
        source "students2"
        target "answers_math_std"
    ]
    edge [
        source "students2"
        target "clsize_snv"
    ]
    edge [
        source "students:segment"
        target "clsize_snv"
    ]
    edge [
        source "students2:segment"
        target "clsize_snv"
    ]
    edge [
        source "m_mom_occ"
        target "clsize_snv"
    ]
    edge [
        source "m_mom_occ"
        target "answers_math_std"
    ]
    edge [
        source "m_mom_edu"
        target "m_mom_occ"
    ]
    edge [
        source "m_mom_edu"
        target "mom_employed"
    ]
    edge [
        source "m_mom_edu"
        target "answers_math_std"
    ]
    edge [
        source "m_mom_edu"
        target "clsize_snv"
    ]
    edge [
        source "mom_housew"
        target "mom_unemp"
    ]
    edge [
        source "m_mom_edu"
        target "mom_unemp"
    ]
    edge [
        source "mom_unemp"
        target "mom_employed"
    ]
    edge [
        source "students:segment"
        target "answers_math_std"
    ]
    edge [
        source "students2:segment"
        target "answers_math_std"
    ]
    edge [
        source "students"
        target "o_math"
    ]
    edge [
        source "segment"
        target "o_math"
    ]
    edge [
        source "enrol_ins_snv"
        target "o_math"
    ]
    edge [
        source "m_mom_edu"
        target "o_math"
    ]
    edge [
        source "dad_lowedu"
        target "o_math"
    ]
    edge [
        source "dad_midedu"
        target "o_math"
    ]
    edge [
        source "dad_highedu"
        target "o_math"
    ]
    edge [
        source "grade"
        target "o_math"
    ]
    edge [
        source "o_math"
        target "answers_math_std"
    ]
    edge [
        source "m_dad_edu"
        target "o_math"
    ]
    edge [
        source "mom_employed"
        target "o_math"
    ]
    edge [
        source "mom_unemp"
        target "o_math"
    ]
    edge [
        source "students2"
        target "o_math"
    ]
    edge [
        source "region"
        target "o_math"
    ]
    edge [
        source "students2:segment"
        target "o_math"
    ]
    edge [
        source "students:segment"
        target "o_math"
    ]
    edge [
        source "interaction_effect_eg_clsize"
        target "clsize_snv"
    ]
    edge [
        source "interaction_effect_eg_clsize"
        target "answers_math_std"
    ]
    edge [
        source "dad_lowedu"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "dad_midedu"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "dad_highedu"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "m_mom_edu"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "m_dad_edu"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "students"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "enrol_ins_snv"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "region"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "segment"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "students2:segment"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "students:segment"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "dad_lowedu"
        target "edu_gap"
    ]
    edge [
        source "dad_midedu"
        target "edu_gap"
    ]
    edge [
        source "dad_highedu"
        target "edu_gap"
    ]
    edge [
        source "m_dad_edu"
        target "edu_gap"
    ]
    edge [
        source "m_mom_edu"
        target "edu_gap"
    ]
    edge [
        source "edu_gap"
        target "interaction_effect_eg_clsize"
    ]
    edge [
        source "edu_gap"
        target "answers_math_std"
    ]
    edge [
        source "immigrants_broad"
        target "clsize_snv"
    ]
]
"""
# This subsection of the DAG section of code was assisted by the report-week6.pdf by Prof. Khazra
# Khazra, N., 2024. report-week6.pdf, Lecture Material,.

import networkx as nx

plt.clf()
plt.figure(figsize=(15, 15))
new_graph = nx.parse_gml(gml_graph)
nx.draw(
    new_graph,
    pos=nx.spring_layout(new_graph),
    with_labels=True,
    node_color="skyblue",
    node_size=1000,
    font_size=6,
    arrows=True,
    arrowsize=10,
    font_color="black",
    edge_color="grey"
)
plt.title("Structural Causal Model")
plt.savefig("SCM")

backdoors = ['m_mom_edu', 'm_mom_occ', 'female', 'm_female', 'grade', 'region', 'enrol_ins_snv', 'survey', 'segment',
             'students', 'students2', 'students2:segment', 'students:segment', 'interaction_effect_eg_clsize']

model = CausalModel(
    data=DAG_x_math,
    treatment='clsize_snv',
    outcome='answers_math_std',
    instruments=['clsize_hat', 'o_math'],
    common_causes=backdoors,
    graph=gml_graph
)
model.view_model()

print(DAG_x_math.columns)

print("SPLIT IDENTIFY ESTIMAND")

# identify the estimand
estimand = model.identify_effect()
print(estimand)

# week 8 report stuff

# IPW weighting
estimate_8 = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.propensity_score_weighting',
    target_units='ate'
)

print('propensity_score_weighting')
print(estimate_8.value)

print('estimate_value')
fart = (estimate_8.value - 10e3) / 10e3
print(fart)

# POSITIVITY GRAPH
import graphviz as gr
import seaborn as sns
from matplotlib import style
from matplotlib import pyplot as plt

sns.distplot(DAG_x_math.query("clsize_snv == 0")["propensity_score"], kde=False, label="Control")
sns.distplot(DAG_x_math.query("clsize_snv == 1")["propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check: Propensity Score Distribution")
plt.xlabel("Propensity Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("positivity_check.png")

# META LEARNER START

from copy import deepcopy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
import dowhy
from dowhy import CausalModel
import dowhy.datasets
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, LGBMClassifier
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
import graphviz
import warnings

warnings.filterwarnings('ignore')

train_data, test_data = train_test_split(DAG_x_math, test_size=0.1, random_state=42)
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

model_train = CausalModel(
    data=train_data,
    treatment='clsize_snv',
    outcome='answers_math_std',
    instruments=['clsize_hat', 'o_math'],
    common_causes=backdoors,
    graph=gml_graph
)
model_train.view_model()

print("SPLIT IDENTIFY ESTIMAND")

# identify the estimand
estimand_train = model_train.identify_effect()
print(estimand_train)

# S-Learner
estimate_train = model_train.estimate_effect(
    identified_estimand=estimand_train,
    method_name='iv.econml.metalearners.SLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
        },
        'fit_params': {}
    })

print('ANSWER')
print(estimate_train)

refutation_train = model_train.refute_estimate(
    estimand=estimand_train,
    estimate=estimate_train,
    method_name='bootstrap_refuter',
    method_params={'num_simulations': 100}
)
print('refutation_train')
print(refutation_train)

print('ANSWER')
print(estimate_train)

# Compute predictions
effect_pred = estimate_train.value
# Get the true effect
effect_true = test_data['answers_math_std'].values
effect_pred_array = np.full_like(effect_true, effect_pred)
# Compute the error
mape = mean_absolute_percentage_error(effect_true, effect_pred_array)
print('mean_absolute_percentage_error')
print(mape)

# T-LEARNER

estimate_T = model_train.estimate_effect(
    identified_estimand=estimand_train,
    method_name='iv.econml.metalearners.TLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': [
                LGBMRegressor(n_estimators=200, max_depth=10),
                LGBMRegressor(n_estimators=200, max_depth=10)
            ]
        },
        'fit_params': {}
    })

print('ANSWER')
print(estimate_T)

# Compute predictions
effect_pred_T = estimate_T.value
# Get the true effect
effect_true_T = test_data['answers_math_std'].values
effect_pred_array_T = np.full_like(effect_true_T, effect_pred_T)
# Compute the error
mape_T = mean_absolute_percentage_error(effect_true_T, effect_pred_array_T)

print('mean_absolute_percentage_error_T')
print(mape_T)

# X LEARNER

estimate_X = model_train.estimate_effect(
    identified_estimand=estimand_train,
    method_name='iv.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': LGBMRegressor(n_estimators=50, max_depth=10),
        },
        'fit_params': {},
    })

print('ANSWER')
print(estimate_X)

# Compute predictions
effect_pred_X = estimate_X.value
# Get the true effect
effect_true_X = test_data['answers_math_std'].values
effect_pred_array_X = np.full_like(effect_true_X, effect_pred_X)
# Compute the error
mape_X = mean_absolute_percentage_error(effect_true_X, effect_pred_array_X)

print('mean_absolute_percentage_error_X')
print(mape_X)

from docx import Document
from docx.shared import Inches

document = Document()

# END WEEK 8

# END META LEARNERS WEEK 8