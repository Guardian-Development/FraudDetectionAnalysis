# Fraud Detection Analysis
Analysing and predicting fraudulent transactions. 

## Initial Data Analysis 
The first step I took after I initially cleaned the data, which once evaluating the data set resulted in the scaling of the amount column. 
I then had a look at column correlations to see if there was anything interesting in the data set before starting predictions. 
![Alt text](/analysis/results/field_correlations_heat_map.png?raw=true)
The heat map shown that some correlations do seem to exist. 
But this becomes more apparent when looking at a cluster map of the same data. 
![Alt text](/analysis/results/field_correlations_cluster_map.png?raw=true)

## Data setup
The program at present reads in the csv provided and then looks at the predicting class distribution. If we have a class
that has less than 10% distribution in the data, then we create a subset of the data frame that contains an even distribution of 
the classes. We then use this smaller data set to train our models, and then use the full data set when predicting. This 
does mean we have some of our training data in our test data, however this allows us to train a more accurate model 
without further reducing the distribution of the classes in the test data. 

## Making Predictions
I have constructed a PredictiveModelsAccessor which can be used to make predictions across multiple configured models. We then can 
either take the most common prediction, or take the prediction that the most accurate models predicted. This has resulted in 
the below findings when using this with configured LogisticRegression, RandomForest and SupportVectorMachines. 

### Accuracy 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 0.97   | 0.98     | 284315  |
| 1      | 0.05      | 0.99   | 0.10     | 492     |
| Total  | 1.00      | 0.97   | 0.98     | 284807  |

### Predictions 
![Alt text](/analysis/results/latest_result.png?raw=true)

# Data set 
The data set used can be found here: https://www.kaggle.com/dalpozz/creditcardfraud/ 

# Next steps 
I will attempt to run this analysis suite against other data sets that do binary classification and see how it performs. 




