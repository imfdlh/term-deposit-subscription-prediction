# Term Deposit Subscription Prediction
A model for predicting whether a client subscribes to a term deposit.

# Deployment Link
https://fadilah-milestone1p1.herokuapp.com/

Template by Fadilah, powered by Bootstrap v5.1.

# Data Source
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

Source: UCI ML Repo > <a href="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing">Bank Marketing Data Set</a>

# Objective
Create a model for predicting whether a client subscribes to a term deposit.

The main objective can be broken down into 4:
1. Compare the performance between each algorithm as follows:
    - Logistic Regression,
    - Support Vector Machine,
    - Decision Tree,
    - Random Forest,
    - K-Nearest Neighbors,
    - Naive Bayes,
    - AdaBoost.
2. Decide which algorithm that we will use based on the best performance on the chosen metrics.
3. Analyze which part of the model needs to be improved based on overall modeling stages.
4. Predict an input of a new client based on the deployed site.

# EDA
**Numerical Features**
- Customer who subscribed has a greater duration during the last contact by the bank which is approximately ~449 seconds or ~7.5 minutes. Customer who didn't subscribe has a lower duration which is ~165 seconds or ~2.7 minutes. This is consistent with the dataset description of the duration variable where those who have duration 0 didn't subscribe to the term deposit.
- Customer who subscribed and didn't subscribe has the same median number of contacts performed during this campaign. But we also can conclude that the maximum number of contacts performed for the customer who ends up subscribed is lower.
- The pdays features are the number of days that passed by after the client was last contacted from a previous campaign. As stated in the dataset description, this feature has an extremely high value at the positive value range (999) that indicates that the client was not previously contacted. This affected the whole distribution, we can't see the real median for those who have been contacted who end up subscribed/didn't subscribe.
    - After excluding those who were not previously contacted, we can see that the median of days that passed by after the client was last contacted is approximately the same which is ~6 days. But the range is lower for those who have subscribed, according to the quartile, for those who subscribed have pdays around 3-6 days those who didn't have pdays around 3-9 days.
- Emp_var_rate (employment variation rate/employment rate dispersion) is the coefficient of variation of regional employment rates in a country, weighted by the absolute population (active population) of each region that quarterly aggregated. The emp_var_rate for those who subscribed are falls within the range of -1.8 to -0.1 or approximately has a median of -1.8 which is smaller than for those who didn't subscribe.
- cons_price_idx is a consumer price index that is aggregated monthly. It measures the average change in prices over time that consumers pay for a basket of goods and services. This feature has a small range of value, the majority falls within the range of 92.8 to 92.4. The median of those who decided to subscribe is slightly smaller than the median who didn't subscribe.
- cons_conf_idx is a consumer confidence index that is aggregated monthly. It measures how optimistic or pessimistic consumers are regarding their expected financial situation. The majority falls within the range of -20.9 to . The median of those who decided to subscribe is slightly greater than the median who didn't subscribe.
- euribor3m (Euro Interbank Offered Rate) is the Euribor rates are based on the average interest rates at which a large panel of European banks borrow funds from one another. This feature is aggregation of 3 month rate. This feature range is really differs compared to the others attributes that contains social and economic information. The majority falls within the range of 0.634 to 50.45. The median of those who decided to subscribe is slightly smaller than the median who didn't subscribe.

**Categorical Features**
- The job features has a variety of ratio between its category. There is a student and retired who seems to have a high ratio that subscribed to the product. After cross-check with the absolute value in the previous plot, we found out that the total amount of people who falls into this category is low compared to other categories that have a greater total of people within groups.
- For the education features, those who have the top 2 highest ratios are also coming from categories that have less total of people within the group. Unlike the previous top 2 ratio, the 3rd-5th highest ratio is coming from categories that have a greater total of people within its group.
- The majority of people who don't have default credit, have the highest ratio of people who subscribed to the term deposit.
- People who have housing loans have a higher ratio of people who subscribed.
- Majority of people who have been contacted on their cellphone are more likely to subscribe.
- People who have been contacted in march, december, september, and october seem to have a higher ratio of people who subscribed to the term deposit. But in fact, we found out that the total number of people who are contacted during those months is really low.
- Those people who have previously succeeded to be converted from the previous marketing campaign, are more likely to subscribe again.

# Model Analysis
Best Model based on AUC: **SVM**

- Based on the AUC and F1 score, SVM Classifier has the best performance compared to other algorithms. The train test is slightly greater than the test, but it still falls within the range of ~73% - 76%.
- We also know that SVM also has the highest recall compared to other models, both on the val and train set. It means this model can capture more actual class 1 than the other models.
- SVM takes a long time to predict values, it's the 2nd slowest model in this experiment. This is due to the nature of the algorithm that involved calculating distances or more precisely, calculates the margin between vectors. The model that uses distance calculation normally takes a longer time. We can see that the bottom 2 times elapsed are SVM and KNN which also use distance measurement.
- As for the highest score of precision, we can see that is achieved by Random Forest Classifier which has precision on val: 67% and on the train: 82%. It means this model, on the train set, can be 82% accurate in predicting class 1.
- Besides the best model (SVM), we can see that the tree-family models have the 2nd-4th best AUC score on val. This might be the nature of our data that not much of the features contain linear relationships. Tree-based models are suitable for handling non-linear data.
- The AUC scores of AdaBoost are quite consistent for the val and train set, this also happened in Gaussian Naive Bayes.
- Gaussian Naive Bayes is the most consistent for all the evaluation metrics scoring on the val and train set.

To achieve greater scores on AUC, and also increase the recall and precision, we can try to use another method of encoding. We also can try to transform the outliers.
