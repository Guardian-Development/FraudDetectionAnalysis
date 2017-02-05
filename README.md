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

### Attempt 1 
#### Analysis on findings 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 1.00   | 1.00     | 113729  |
| 1      | 0.69      | 0.63   | 0.66     | 194     |
| Total  | 1.00      | 1.00   | 1.00     | 113923  |

As you can see a lot of this data is fairly useless, as such a small proportion of our data is in class 1. However, 
we can take away from this that we are between 60% and 70% accuracy when predicting fraudulent transactions. 
#### Confusion matrix
![Alt text](/analysis/results/logistic_regression_attempt_1_heat_map.png?raw=true) </ br> 
As you can see we predicted 122 fraudulent transactions successfully, we missed 55 and wrongly labelled 72 genuine transactions.

