# CS506 - Final Project

Members: Ethan Machleder, Keith Yeung, Jason Zhu

## Description

This project aimed to predict the price of a resell car based on a dataset that includes various car features such as make, model, year, condition, odometer reading, and other categorical and numerical attributes. This analysis focuses on feature selection and model tuning of advanced regression techniques, including linear regression, Ridge regression, Lasso regression, Random Forest, and XGBoost models, to determine the most accurate method for price prediction.

## Background

Predicting car prices is a critical task for car dealerships, buyers, and sellers. Accurate predictions can help in pricing cars competitively, understanding market trends, and identifying factors that significantly influence prices. The dataset includes historical sales data, vehicle features, and sales dates. This task involves cleaning and preprocessing the data, exploring feature importance, and evaluating regression models to achieve reliable predictions.

## Motivation and Goals

We want to build a robust model to predict car selling prices. We also wanted to compare the performance of multiple regression models and hope that it provides actionable insights based on the analysis. This is informative for car dealerships and personal sellers, as well as potential buyers to estimate the cost of a car. This would allow people to accurately predict the cost of a car and get the most out of their money.

## Data Preprocessing

Our dataset was obtained from Kaggle. It consisted of 16 variables and 558,838 rows of data. The dataset included the following columns: categorical features such as make, model, trim, body, transmission, state, color, and interior; numerical features such as year, condition, and odometer; temporal features such as saledate; and our target variable, selling price. Other columns such as vin and seller were immediately not useful to our project. The vin is unique to every car and has no relevance to predicting the selling price. The seller could be useful but there were simply too many unique sellers outside of the official brands of carmakers. Furthermore, the make was already a variable. 

The first step in our data preprocessing was focused on identifying and handling missing values to ensure that our data could be used fully. Initially, we examined each column to quantify the number and percentage of missing values, giving us a clear overview of data completeness. Based on this assessment, we identified essential columns such as make, model, trim, body, transmission, and condition that were missing over 10,000 rows. After further investigation, we found that most of the data values that were missing one variable, were also missing the other variables. Thus, to retain data quality, we chose to drop all the rows where these key features were missing. After this cleaning step, there were still 472,338 rows of data remaining. 

Our next step was to further investigate each of our variables. We calculated the frequency of unique values in each of our categorical variables like make, model, trim, body, color, and interior and plotted each one. We found that there were 40 different makes in our dataset; the largest five being Ford, Chevrolet, Nissan, Toyota, and Dodge. This makes sense, as these are the most popular and affordable car brands. There were very few cars from super luxury brands like Maserati, Bentley, Aston Martin, Ferrari, Rolls-Royce, and Lamborghini. There were simply way too many unique models and trims as each brand had many models and each model had different trims. For car bodies, we simplified the different car bodies into more general groups. For example, ‘Cab’, ‘Crew cab’, ‘Extended Cab’, ‘Regular Cab’, and ‘Quad Cab’ were grouped together as ‘Cab’. This reduced the number of unique car bodies so it was more manageable to work with. As for color and interior color of the car, both had very skewed frequencies, with the majority being common colors such as black or white, but there were a few rare colors such as pink, turquoise, etc. We decided that such a low sample (< 0.2% of entire dataset) for each other would not be helpful and accurate in predicting price anyway so we grouped them together as ‘Other’. 

































## Proposal:

Many different factors affect the price of a car, such as the brand, mileage, manufacturing year, and features of the car. For our project, we want to be able to predict the price of cars based on these variables successfully.

The data that needs to be collected is information on car sales and as many features of the cars as possible. We may be able to find this data through past car auctions, most likely from Kaggle datasets.

We plan on modeling the data by fitting a linear model to model the relationship between certain features and prices. To model more complex relationships between multiple features and prices, we can create decision trees or implement an XGBoost algorithm.

For basic descriptive visualizations, we will have bar graphs to compare different features as well as a correlation heatmap. We will also have interactive scatterplots so users can explore how different features impact car prices.

We plan to split the collected data into training and testing sets, using 70-80% of the data for training and 20-30% for testing. We will implement cross-validation techniques to prevent overfitting and validate our models. Model performance will be evaluated using metrics such as Mean Squared Error (MSE).

## Report:

### Video: https://youtu.be/wwNJvamMoh8

Our preliminary data processing focused on identifying and handling missing values to ensure data integrity. Initially, we examined each column to quantify the number and percentage of missing values, giving us a clear overview of data completeness. Based on this assessment, we identified essential columns with significant gaps, such as: make, model, trim, body, odometer(mileage), color, interior, transmission, and condition.

To retain data quality, we chose to drop rows where these key features were missing, as they represent critical information for our analysis. After this cleaning step, we confirmed the reduction in missing data by recalculating the number and percentage of null values, leading to a refined dataset that better supports accurate modeling and analysis.

We calculated the frequency of unique values for each column, providing a clearer picture of the most and least represented categories. These counts offer a foundational understanding of our dataset's composition and can inform further analysis, such as identifying popular car makes or common body styles. We also standardized the text format in the body column by converting each entry to a title case, ensuring consistency across values (e.g., converting "sedan" to "Sedan"). This step minimizes potential issues in downstream analyses caused by inconsistent text formats. Additionally, we saved the sale date column and the calculated frequency counts to text files to maintain a record of these key attributes for reference and ease of access in future analyses.

To better understand our dataset, we used various graphs to reveal trends, distributions, and relationships within key features. By plotting the frequency counts for categorical columns like make, model, trim, and body, we gained a clearer view of the distribution of car types, popular models, and common trim levels.
The box plots allowed us to examine the distribution and spread of selling prices within each category, revealing how specific attributes may impact vehicle pricing. Overall, these visualizations provide valuable insights into both numerical relationships and categorical trends, guiding us in selecting relevant features for further modeling.

To evaluate our model’s performance, we made predictions on the test set (X_test) and compared them to the actual values (y_test). We calculated key evaluation metrics, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared (R²) score. MSE provides the average squared difference between predicted and actual values, while RMSE, as the square root of MSE, indicates the model’s prediction error in the same units as the target variable. The R² score shows the proportion of variance in the selling price explained by our model, giving an overall measure of model fit. These metrics help us assess prediction accuracy and model effectiveness.
