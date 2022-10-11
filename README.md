# DPS Challenge

An AI model based on XGBoost Regressor has been trained on the "Monatszahlen Verkehrsunfälle" Dataset to predict the value using only Year and Month parameters. 

Since the dataset is a multivariate dataset that consisted of many data points with the same year and month but different "Category" and "Type",
separate models have been built for separate categories (3 have been considered).

According to the request, the specific model is used to predict.

For validation, values of 2020 have been used whereas for testing, values of 2021 have been used. Forecasts have also been plotted.
From the plots, we can see that XGBoost gives the best result for the "Alkoholunfälle" Category. One reason for the results not being that good is less data. Only
monthly data is given.

