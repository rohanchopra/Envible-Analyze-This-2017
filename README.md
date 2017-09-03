# Team Envible Analyze This 2017
Repository for American Express [Analyze This 2017](https://in.axpcampus.com/AnalyzeThis/competition.php) challenge
## Competition
The aim of the competition was to offer the students a pedagogical and professional experience in the analytics industry and provide an opportunity to explore analytics from a practical perspective.

## Data
The dataset contained user data and we had to predict
1) If the user will buy a credit card
2) And which credit card will he buy 

## Approach
* Our approach was to first perform binary (buying a card or not) classification and then multiclass (buy which card) classification. 
* The reasoning for the approach taken was that the data was highly skewed. The ratio of users **not buying** a card to buying **category 1** card to **category 2** card to **category 3** card was **30 : 3.5 : 2.8 : 2.7**.
* This two step approach helped us takle the problem of skewed data to some extent.

## Problems faced
* Highly skewed data
* We were not able to perform predictions on the XGboost model due to a bug in the H2O framework and time constraint
* Limited time to use all the data analysis findings in the machine learning model building

## Result
* Our final logloss on cross-validation using GBM was around 0.73
* Using XGboost for multiclass classification logloss was around 0.71
* The final submission made use of GBM for both binary(card or no card) and multiclass(which card) classification