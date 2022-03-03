#!/usr/bin/env python
# coding: utf-8

# Survival analysis is an important concept with lots of relevance in Business
# 
# Some examples of application include:
# - SaaS providers are interested in measuring subscriber lifetimes, or time to some first action
# - inventory stock out is a censoring event for true "demand" of a good.
# - sociologists are interested in measuring political parties' lifetimes, or relationships, or marriages
# - A/B tests to determine how long it takes different groups to perform an action.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve 
from sklearn.metrics import brier_score_loss
#! pip install lifelines==0.26.4
get_ipython().system('pip install lifelines')
import lifelines


# In[16]:


data = pd.read_csv('/Users/school/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[17]:


pd.set_option('display.max_columns', None)
data.head()


# In[18]:


# we see that the 488 row is a space
data.loc[488,'TotalCharges']


# In[19]:


# Save customerID and MonthlyCharges columns in a separate DF and drop customerID from the main DF
churned_customers = data[data['Churn'] == 'No']
customerID = pd.DataFrame(churned_customers[['customerID', 'MonthlyCharges']])
data.drop(columns = ['customerID' , 'TotalCharges'], inplace=True)



# Replace single whitespace with MonthlyCharges and convert to numeric
data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'])
data['TotalCharges'].replace(' ', np.nan, inplace=True)
data.dropna(subset=['TotalCharges'], inplace=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])


# We will also create a copy of DF that will be utilised later to plot categorical KM Curves
data_kmf = data.copy()


# In[20]:


# Convert Churn column to 1 (Yes) or 0 (No)
data['Churn'] = data['Churn'].replace({"No": 0, "Yes": 1})


# Note that some features has 3 categories: Yes, No and No phone service, for our purpose we can treat no phone service as just No.

# In[21]:


# Create a list of features where we will assign 1 to a Yes value and 0 otherwise
features_to_combine = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
   'TechSupport', 'StreamingTV', 'StreamingMovies']

# Assign 1 to Yes and 0 otherwise
for feat in features_to_combine:
    data[feat] = data[feat].apply(lambda x: 1 if x == 'Yes' else 0)


# In[22]:


data.head()


# We can create dummy variables for the remaining categorical variables like Phone service, Contract...
# 
# Note that in the pd.get_dummies, we set drop_first to False. This change is intentional so that we can mannually control what we want to set as the baseline model. This choice is made for better understanding of the business case and model
# 

# In[23]:


data = pd.get_dummies(data, columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                     'InternetService', 'Contract', 'PaperlessBilling',
                        'PaymentMethod'], drop_first = False)


# In[24]:


# Drop that dummy variable that the business considers to be typical of their subscribers
data.drop(columns = ['gender_Male', 'Partner_Yes', 'Dependents_No', 'PhoneService_Yes',
   'InternetService_Fiber optic', 'Contract_Month-to-month', 'PaperlessBilling_Yes',
    'PaymentMethod_Electronic check'], inplace = True)


# In[25]:


data.head()


# Let's first visualize the continous variables between churned and non churned customers

# In[12]:


sns.set_style('whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20,6))
fig.suptitle("Kernel Density Function of continuous feature variables")
# tenure
sns.kdeplot(data['tenure'].loc[data['Churn'] == 0], label='not churn', shade=True, ax = ax1, color = 'green')
sns.kdeplot(data['tenure'].loc[data['Churn'] == 1], label='churn', shade=True, ax = ax1, color = 'red')
ax1.set_xlabel("Tenure (Months)")
# monthly charges
sns.kdeplot(data['MonthlyCharges'].loc[data['Churn'] == 0], label='not churn', shade=True, ax = ax2,color = 'green')
sns.kdeplot(data['MonthlyCharges'].loc[data['Churn'] == 1], label='churn', shade=True, ax = ax2,color = 'red')
ax2.set_xlabel("Monthly Charges ($)")
# total charges
sns.kdeplot(data['TotalCharges'].loc[data['Churn'] == 0], label='not churn', shade=True, ax = ax3,color = 'green')
sns.kdeplot(data['TotalCharges'].loc[data['Churn'] == 1], label='churn', shade=True, ax = ax3,color = 'red')
ax3.set_xlabel("Total Charges ($)");


# We see from the first plot 
# - that the churned customers are usually those who are new to the service
# - It could be that these customers change service provider frequently to negogiate better deal
# 
# We see from the second plot 
# - those who churn tend to have higher charges
# 
# From the third plot
# - we see similar phenomenon in the first plot, this might be due to total charges is a proxy with tenure

# In[14]:


# Let's check for multicollinearity as it is a basic assumption of the CPH model
fig = plt.figure(figsize=(10,10))
corrmat = data.corr()
sns.heatmap(corrmat);


# In[ ]:





# Setting up data for futher analysis

# In[26]:


# Update Churn column of data_kmf that we kept aside for this moment before feature engineering
data_kmf['Churn'] = data_kmf['Churn'].replace({"No": 0, "Yes": 1})


# In[27]:


# save indices for each contract type
idx_m2m = data_kmf['Contract'] == 'Month-to-month'
idx_1y = data_kmf['Contract'] == 'One year'
idx_2y = data_kmf['Contract'] == 'Two year'


# In[28]:



T1 = data_kmf.loc[idx_m2m, 'tenure']
T2 = data_kmf.loc[idx_1y, 'tenure']
T3 = data_kmf.loc[idx_2y, 'tenure']
E1 = data_kmf.loc[idx_m2m, 'Churn']
E2 = data_kmf.loc[idx_1y, 'Churn']
E3 = data_kmf.loc[idx_2y, 'Churn']



# # The Kaplan _Meier Survival Curve

# The survival curve or survival function is defined as 
# $$ S(t) = Pr(T > t) $$
# 
# To exemplify the equation, in this example, the company is interested in modelling customer churn. Let T measure the time that a customer cancels their subscription. At any given time T = t, S(t) will be the probability that a customer cancels later time t. The higher the probability, the less likely the customer has churned before time t.
# 
# However as with most things in Statistics, we do not know the true statistical model and hence the best thing we can do is have a good estimation of the true distribution.
# 
# The Kaplan Meier is a method to estimate the survival curve.
# 
# 
# For the purpose of the business case, I suspect that the contract type plays a major factor in predicting churn of a customer. We will group the customers by their contract type: 3 months, 1 year and 2 years respectively.
# 
# I have ommited the math but interested readers should go to "An introduction to Statistical Learning" Chp 11 by James, Hastie, Tibshirani et.al

# In[29]:


# Update Churn column of data_kmf that we kept aside for this moment before feature engineering
data_kmf['Churn'] = data_kmf['Churn'].replace({"No": 0, "Yes": 1})

# plot the 3 KM plots for each category
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))
kmf_m2m = lifelines.KaplanMeierFitter()
ax = kmf_m2m.fit(durations = T1, event_observed = E1, label = 'Month-to-month').plot(ax = ax)
kmf_1y = lifelines.KaplanMeierFitter()
ax = kmf_1y.fit(durations = T2, event_observed = E2, label = 'One year').plot(ax = ax)
kmf_2y = lifelines.KaplanMeierFitter()
ax = kmf_2y.fit(durations = T3, event_observed = E3, label = 'Two year').plot(ax = ax)

# display title and labels
ax.set_title('KM Survival Curve by Contract Duration')
ax.set_xlabel('Customer Tenure (Months)')
ax.set_ylabel('Customer Survival Chance')
plt.grid()

# display at-risk counts for each category
lifelines.plotting.add_at_risk_counts(kmf_m2m, kmf_1y, kmf_2y, ax = ax);


# From the Kaplan Meier estimation of the survival curve, it seems like customers who are on a month to month contract churns the most, one year is in the middle while 2 year contracts really locks in the customer and we can see that by the end of 1 year, most of the 2-yeared contract customers are still with the company.
# 
# We have also incorporated the 5% standard error bands for each survival curve and we can reasonably say that the three groups have different survival curve without doing further work.
# 
# However, what if we want to have a formal test of equality of the 3 survival curves ? We would need the Log-Rank test.
# 

# # Log-Rank test

# To be sucinct, Log-rank test is borrows idea from 2-sample t-test, but due to the censoring nature of the observations,we have to somehow incorporate information as to how the events in each group unfold sequentially over time.

# In[30]:


from lifelines.statistics import logrank_test
results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)


# In[31]:


print(len(T1),len(T2),len(T3))


# In[32]:


results.print_summary()
print(results.p_value)        
print(results.test_statistic)


# In[ ]:


#from lifelines.statistics import multivariate_logrank_test


# Since the p-value is less than 0.005 we reject the Null hypothesis. This suggests there is evidence against 
# the 2 contract types ignoring other factors. Those with a month to month seems to have shorter survival time 
# than the 1 year contract, ignoring other factors.
# 
# However there are several limitations:
# 
# 1) we can't test multiple hypothesis at the same time 
# 
# 2) we simply can't account for confounders or have continous variable. 
# 
# 3) Also since KM is non-parametric we can't summarize the relationship with a number/ coefficient. We would require
# the below graph to express the relationship.
# 
# KM is only a good initial starting point, to really address for confounders and to create something with
# more predictive power we move next to the CPHM

# # Cox Proportional Hazards Model with cross validated model

# Quick primer:
# its another type of regression model, we can incorporate numeric explanatory variables, non parametric, doesn't assume constant hazard
# 
# recall the harzard means the probability you die now given you are alive, for practicaliy you can think of it as the probability you will die the next milisecond. It is not really a useful concept on its own, however it is used in the mathematical formulation.
# 
# Cox has log base line hazard, unspecified, it is allowed to vary over time, note that we also do not know what the base line hazard is so this model can't be used to predict someone's hazard at a particular time , like you would want to use a linear regression model to predict Y.
# 
# what we can do is estimate the hazard ratio, to compare between 2 groups.

# In[34]:


# Instantiate and fit CPH model
cph = lifelines.CoxPHFitter(alpha= 0.01)
cph.fit(data, duration_col = 'tenure', event_col = 'Churn')

# Print model summary
cph.print_summary(model = 'base model', decimals = 3, columns = ['coef', 'exp(coef)', 'p']) 

# the Cross validated goodness of fit:
#k_fold_cross_validation(cph, data, duration_col='tenure', event_col='Churn', scoring_method="concordance_index")
lifelines.utils.k_fold_cross_validation(cph, data, 'tenure', 'Churn', k = 10, scoring_method = 'concordance_index')


# # Interpreting the model

# the column 'coef' is the coefficient of each variable on risk.
# 
# The higher the coef, the more likely the feature is to contribute to churn
# 
# the column 'exp(coef)' is the hazard ratio. Note that this ratio is a comparison between 2 groups. The numerator represents the group of interest, while the demonominator represents the baseline group. For example if the hazard ratio for 'InternetService_N' is 0.035, this implies a customer who does not have internet service is 0.035 as likely to cancel their subscription compared to the base line (with internet service)

# In[35]:


# Plotting the coefficients
fig_coef, ax_coef = plt.subplots(figsize = (20,10))

ax_coef.set_title('Coefficients and 99% Confidence Intervals of fitted Model', fontsize=30)
cph.plot(ax = ax_coef);


# The above plot shows which characterisitics are likely to be correlated to churn is 
# 1) having no Partner
# 
# while those who are not likely to churn:
# - contract_two year
# - contract_one year
# - automatic bank transfer payment
# - using credit card to pay
# - using paper billing
# 
# Note that InternetService No has a really large confidence interval hence we retain from including it in the list
# 

# # Monthly charges effect on survival

# Here we can plot the partial effect of monthly charges on the outcome of the survival curve. It is not suprising to see that customers with higher monthly charges tends to churn more than those with lower charges.
# 
# There will be an interesting optimization problem as to setting the correct monthly prices for each specific customer to maximize revenue. 
# 

# In[96]:


cph.plot_partial_effects_on_outcome(covariates='MonthlyCharges', values=[10, 20,30,40,50,60,70,80,90,100], cmap='coolwarm')


# # Prediction:

# Predict the survival function for individuals, given their covariates. This assumes that the individual just entered the study (that is, we do not condition on how long they have already lived for.)

# In[97]:


# first filter for 'alive' customers
censored_data = data[data['Churn'] == 0]
# remove the tenure column as we do not want to condition on this variable
censored_data_last_obs = censored_data['tenure']
# Predict the survival function for each customer using the explanatory variables, from this time point on
conditioned_sf = cph.predict_survival_function(censored_data, conditional_after = censored_data_last_obs)
conditioned_sf


# In the above table, the 72 rows, represent the number of months from today's time, while the columns represent each individual customers who are still with the company, as you see for each customer, as the month passes, it is more likely for them to churn and leave the company.

# Note that if we want to interpret on what month will a customer cancel their plans with the company, we need to set a threshold where if the probability passing the threshold implies the customer has churned. 
# 
# Out of all the thresholds, the median would be the more robust to outlier and is a representative attribute for this business context.
# 
# For now, let's set the threshold to the median

# In[100]:


pred_50 = lifelines.utils.qth_survival_times(0.5, conditioned_sf)
pred_50


# With that the company can more concretly pinpoint when each customer is going to churn, this leads to a lot of future business strategies for customer retention. 
# 
# Note that the management can also play with the quantile, we can set it to the 90th quantile as a threshold if we want to get a 'warning' sign that a customer is thinking of leaving. There really isn't much you can do/ there will already be irreversible loss if you wait till the customer have already left before doing anything.

# In[ ]:




