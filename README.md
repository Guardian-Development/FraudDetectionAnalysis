# Fraud Detection Analysis
Analysing and predicting fraudulent transactions. 

## Initial Data Analysis 
The first step I took after I initially cleaned the data, which once evaluating the data set resulted in the scaling of the amount column. 
I then had a look at column correlations to see if there was anything interesting in the data set before starting predictions. 
![Alt text](/analysis/results/field_correlations_heat_map.png?raw=true)
The heat map shown that some correlations do seem to exist. 
But this becomes more apparent when looking at a cluster map of the same data. 
![Alt text](/analysis/results/field_correlations_cluster_map.png?raw=true)

## Making Predictions
I have then constructed a PredictiveModelsAccessor which can be used to make predictions across multiple configured models. We then can 
either take the most common prediction, or take the prediction that the most accurate models predicted. This has resulted in 
the below findings when using this with LogisticRegression, RandomForest and SupportVectorMachines. 

### Accuracy 
| Class  | Precision | Recall | f1-score | support |
|--------|-----------|--------|----------|---------|
| 0      | 1.00      | 0.99   | 0.99     | 113729  |
| 1      | 0.13      | 0.94   | 0.23     | 194     |
| Total  | 1.00      | 0.99   | 0.99     | 113923  |

### Predictions 
![Alt text](/analysis/results/05:03PM_February_11_2017_results.png?raw=true)

##Project Structure 




