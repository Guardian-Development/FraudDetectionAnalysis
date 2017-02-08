# Fraud Detection Analysis
Analysing and predicting fraudulent transactions 

## Initial Data Analysis 
The first step I took after I initially cleaned the data, 
which only took the scaling of the amount column, 
was to look for any correlations in the fields. 
![Alt text](/analysis/results/field_correlations_heat_map.png?raw=true)
The heat map shown that some correlations do seem to exist. 
But this becomes more apparent when looking at a cluster map of the same data. 
![Alt text](/analysis/results/field_correlations_cluster_map.png?raw=true)

## Logistic Regression 

### Step 1 
My first look into using basic Logistic Regression to predict the fraudulent transactions. 

#### Analysis on findings 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 1.00   | 1.00     | 113729  |
| 1      | 0.69      | 0.63   | 0.66     | 194     |
| Total  | 1.00      | 1.00   | 1.00     | 113923  |

As you can see a lot of this data is fairly useless, as such a small proportion of our data is in class 1. However, 
we can take away from this that we are between 60% and 70% accuracy when predicting fraudulent transactions. 
As you can see below we predicted 122 fraudulent transactions successfully, we missed 55 and wrongly labelled 72 genuine transactions.
#### Confusion matrix
![Alt text](/analysis/results/logistic_regression_attempt_1_heat_map.png?raw=true) 

### Step 2 
Now I am training the Logistic Regression algorithm on a data set that has an even split of classification points. We can then see that we get 
a much higher accuracy on predicting the fraudulent transactions as we have an evenly distributed data set. 

#### Analysis on findings 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 0.92      | 0.96   | 0.94     | 195     |
| 1      | 0.96      | 0.91   | 0.94     | 199     |
| Total  | 0.94      | 0.94   | 0.94     | 394     |
As you can see we have increased our ability to predict the fraudulent class of data by giving it a higher distribution percentage 
in our training set. 
#### Confusion Matrix
![Alt text](/analysis/results/logistic_regression_attempt_2_heat_map.png?raw=true) 

### Step 3 
Now I am training the Logistic Regression algorithm on the evenly split data set but then running the predictions against the full data set. 
We can see from the results that our overall accuracy has reduced as we are predicting a lot of regular transactions as fraudulent, however
we can see we have successfully predicted more of the fraudulent ones.

#### Analysis on findings 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 0.97   | 0.98     | 113729  |
| 1      | 0.05      | 0.92   | 0.09     | 194     |
| Total  | 1.00      | 0.97   | 0.98     | 113923  |
Analysing this, we can see a good step moving forward will be to look at making the algorithm more stringent on labelling fraudulent transactions  
and I will look at changing some of the setup variables to see what gives us the best result. But this is a significant improvement as only 
16 fraudulent transactions made it past our predictions. 
#### Confusion Matrix
![Alt text](/analysis/results/logistic_regression_attempt_3_heat_map.png?raw=true) 

# Next Steps 
I will now lok at tweaking the performance on the algorithm to give the best results. 


