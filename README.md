# LoanPredictionSpark
Simple Loan Status Prediction Problem solved using Apache Spark-ML library.

## Problem Statement
###### About Company
Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

###### Problem
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.


## Dataset Link

###### Training Set
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/download/train-file

###### Test Set
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/download/test-file

## Solution
For visual analysis of data, I have used Zeppelin. With Spark-SQL commands in zeppelin you can plot data with SQL Statements.

###### Hypothesis
1. People with Credit History and Higher Income would be having more chances of Loan Approval.
2. People with lower EMIs would be more likely to repay the loan and hence should be approved loan easily. 
3. People with businesses are more likely to be approved loan. 
4. People with higher age are more likely to be approved loan. 

###### Data Cleaning
Cleaned data and imputed the missing values in the data. 

###### Feature Dependency Check 
Used ANOVA(Analysis of Variance), Z- Test, T-Test, Chi-Square Test, Correlation to know about the statistical dependency of features.

###### Feature Engineering
1. Created an EMI feature which was a measure of Loan_Amount, Loan_Amount_Term. 
2. Created a feature combining Total Income and credit history. 

###### Model 
Using a simple Logistic Regression Model with a list of regularization parameters and maximum iterations to achieve the maximum accuracy. 

###### Accuracy Achieved 
The accuracy achieved is around 79.16%.
