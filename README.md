# CS506 - Final Project

Members: Ethan Machleder, Keith Yeung, Jason Zhu

## Description

This project aimed to predict the price of a used/resold car based on a dataset that includes various car features such as make, model, year, condition, odometer reading, and other categorical and numerical attributes. This analysis focuses on feature selection and model tuning of advanced regression techniques, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and XGBoost models, to determine the most accurate method for price prediction.

## Background

Predicting car prices is a critical task for car dealerships, buyers, and sellers. Accurate predictions can help in pricing cars competitively, understanding market trends, and identifying factors that significantly influence prices. The dataset includes historical sales data, vehicle features, and sales dates. This task involves cleaning and preprocessing the data, exploring feature importance, and evaluating regression models to achieve reliable predictions.

## Motivation and Goals

We wanted to build a robust model to predict car selling prices. We also wanted to compare the performance of multiple regression models and hope that it provides actionable insights based on the analysis. This is informative for car dealerships and personal sellers, as well as potential buyers to estimate the cost of a car. This would allow people to accurately predict the cost of a car and get the most out of their money.

## Data Preprocessing

Our dataset was obtained from Kaggle and consisted of data mostly collected in 2015 and included 16 variables and 558,838 rows of data. The dataset included the following columns: categorical features such as make, model, trim, body, transmission, state, color, seller, and interior; numerical features such as year, condition, vin, and odometer; temporal features such as saledate; and our target variable, selling price. Some columns such as vin and transmission were immediately not useful to our project. The vin is unique to every car and has no relevance to predicting the selling price. Using the vin to search up the vehicles sales history was also very inconsistent and was ruled out. The transmission variable was 97% automatics, and was determined to not be useful.

![dataset info](https://github.com/user-attachments/assets/99b6e25b-2c84-4c27-b6cb-6eeb4d20d110)

Figure 1. Dataset Information including Columns and Rows

The first step in our data preprocessing was focused on identifying and handling missing values to ensure that our data could be used fully. Initially, we examined each column to quantify the number and percentage of missing values, giving us a clear overview of data completeness. Based on this assessment, we identified essential columns such as make, model, trim, body, transmission, and condition that were missing over 10,000 rows. After further investigation, we found that most of the data values that were missing one variable, were also missing the other variables. Thus, to retain data quality, we chose to drop all the rows where these key features were missing. After this cleaning step, there were still 472,338 rows of data remaining. 

![missing values](https://github.com/user-attachments/assets/d3b80fd5-1e43-4b4d-a71f-c553967ed628)

Figure 2. Missing Values from the Dataset

![after cleaning dataset info](https://github.com/user-attachments/assets/0f60d746-9868-4a35-a9c6-ee67ec6fa6dc)

Figure 3. Dataset Information after Cleaning

Our next step was to further investigate each of our variables. We calculated the frequency of unique values in each of our categorical variables like make, model, trim, body, color, and interior and plotted each one. We found that there were 40 different makes in our dataset; the largest five being Ford, Chevrolet, Nissan, Toyota, and Dodge. This makes sense, as these are the most popular and affordable car brands. There were very few cars from super luxury brands like Maserati, Bentley, Aston Martin, Ferrari, Rolls-Royce, and Lamborghini. There were simply way too many unique models and trims as each brand had many models and each model had different trims. For car bodies, we simplified the different car bodies into more general groups. For example, ‘Cab’, ‘Crew cab’, ‘Extended Cab’, ‘Regular Cab’, and ‘Quad Cab’ were grouped together as ‘Cab’. This reduced the number of unique car bodies so it was more manageable to work with. As for color and interior color of the car, both had very skewed frequencies, with the majority being common colors such as black or white, but there were a few rare colors such as pink, turquoise, etc. We decided that such a low sample (< 0.2% of entire dataset) for each other would not be helpful and accurate in predicting price anyway so we grouped them together as ‘Other’ with new variables interior_replaced and color_replaced.

![bar chart of body](https://github.com/user-attachments/assets/35b1ff2d-c02c-4e21-bcb0-ca563d4cc9da)

Figure 4. Bar Chart showing Frequency of Car Bodies before Grouping

We extended our analysis by investigating the temporal aspect of the dataset using the saledate column. After converting it to a standardized datetime format, we identified and logged any invalid dates, ensuring the data's integrity for further analysis. To gain deeper insights, we first tried extracting new temporal features: sale_hour, sale_day, sale_month, sale_year, and sale_day_of_week. We thought these features would be helpful in detecting meaningful correlations and trends, but we later found that there were basically no correlations between the saledate and the price of the car. We believe this is because only the vehicle_age, the number of years since manufacture, was a useful factor. Numeric features like odometer and year displayed trends such as a preference for newer, low-mileage vehicles at affordable price ranges, with selling prices exhibiting a right-skewed distribution. These insights highlight key trends and relationships in the data, informing feature selection for predictive modeling and data preprocessing strategies.

![bar chart of make](https://github.com/user-attachments/assets/fef42be4-ce0c-4683-9ccd-d89bbd7cb102)

Figure 5. Bar Chart of Make showing Frequency of Car Brands

![distribution of year](https://github.com/user-attachments/assets/56e6dead-ca7e-447d-b197-e51a30c087d0)

Figure 6. Histogram of Year showing Distribution of Manufacture Years

We enhanced our analysis by examining relationships between variables and our target variable, sellingprice using scatterplot matrices, heatmaps, and box plots. A scatterplot matrix of numeric features provided a visual exploration of pairwise relationships, helping us identify potential linear or nonlinear patterns that could impact the selling price. The correlation heatmap highlighted the strength and direction of relationships among numeric variables, revealing key predictors of sellingprice and identifying multicollinearity issues that may require attention during modeling. Additionally, box plots of sellingprice against categorical features like make and grouped_body allowed us to visualize how different categories influence price distribution, such as luxury brands consistently exhibiting higher price ranges. These visualizations provide critical insights for feature selection, model tuning, and understanding how various factors contribute to pricing trends in the dataset.

![box plot of color vs price](https://github.com/user-attachments/assets/20cb66a1-431e-4bed-9515-2887329b4bb2)

Figure 7. Box Plot of Color vs Selling Price

![scatterplot to selling price](https://github.com/user-attachments/assets/e5297e38-bb23-4070-a2cf-4a2a67138600)

Figure 8. Scatterplot Matrix to show correlation between features

![heatmap to selling price](https://github.com/user-attachments/assets/d4d6d8ba-ab15-4574-84d3-31b5e3756813)

Figure 9. Correlation Heatmap to show correlation between features

## Feature Engineering

To enhance feature engineering, we introduced several data transformations and groupings. As stated, a new grouped_body feature was created by mapping the body column into broader categories such as Sedan, SUV, and Cab, which simplifies the variety of car body types and makes them more interpretable. We also calculated a vehicle_age feature based on the difference between the sale year and the car’s manufacturing year, which provides a clearer indicator of depreciation trends. Rare colors like charcoal, turquoise, and pink were grouped as Other in the color column, along with less common interior colors such as gold and purple, to address sparsity in these categories. These additions improve data consistency and provide valuable predictors, aiding in better trend analysis and model performance and reduce chances of overfitting.

![post feature adding](https://github.com/user-attachments/assets/712ec7ee-b251-4913-98f2-9e169eb283a3)

Figure 10. Categorical Features after grouping and simplification

To prepare the data for predictive modeling, columns such as vin, trim, saledate, mmr, transmission, interior, color, and body were excluded as they were either unique identifiers, replaced, or redundant for prediction with one exception. The mmr value represents the Manheim Market Report. It is a widely used tool in the automotive industry, particularly by dealers and wholesalers, to estimate the current wholesale market value of a vehicle. We excluded it from the model as even though it gave us R^2 scores of up to 0.95 we figured that it was "cheating" as the mmr score was similar to the selling_price in that it was what we were trying to predict.

The target variable, selling_price, was separated, and the data was split into training and testing sets with a 70-30 ratio. Selected categorical features included make, model, grouped_body, state, color_replaced, interior_replaced, and seller, while numerical features consisted of year, condition, odometer, and vehicle_age. Data preprocessing was handled using a ColumnTransformer, where numeric features were standardized with a StandardScaler and categorical features were encoded with a OneHotEncoder. To evaluate model performance, a visualization function was created to plot actual vs. predicted prices, showing predictions against true values with a diagonal line indicating perfect prediction. This process aids in assessing the model’s accuracy and identifying areas for refinement.


## Modeling

A linear regression model pipeline was created to predict car prices using the preprocessed features. The pipeline consists of two main steps: first, the data is transformed through the preprocessor, which standardizes numeric features and applies one-hot encoding to categorical features; then, a LinearRegression model is fitted to the training data. The model is trained using the training set (X_train, y_train), allowing it to learn the relationships between the features and the target variable, sellingprice. This pipeline was repeated with other regression models, such as Ridge Regression, Lasso Regression, Random Forest, and xgboost. We chose to include other regression models such as Ridge and Lasso to compare the effectiveness of such models and to handle possible multicollinearity and shrink less important features. 

## Results and Interpretation

We chose to measure the effectiveness of our models using root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R^2). We chose these metrics because R^2 indicates the proportion of variance in the selling price that can be explained by the independent features in the model and provides insight as to how well our model fits the data. RMSE measures the square root of the average of the squared differences between predicted and actual values and MAE measures the average of the absolute differences between the predicted and the actual values. We did this as even though we intend to predict the actual selling price, the value we most likely will reach is the mmr, which is not always the same as the selling price, but is very close. These metrics are in the same dollar units as the target variable so it can be easily interpreted. A smaller RMSE and MAE indicates a more accurate model. 

For our Linear Regression model, we got an R^2 of 0.8491, an RMSE of 3735.1426, and an MAE of 2379.8014. This means that approximately 84% of the variability in the selling price can be explained from our independent variables. The high R^2 score indicates the model is effective at predicting car prices based on the provided features. The RMSE (3,735) and MAE (2,380) provide insight into the magnitude of prediction errors, which are reasonably small given the context of selling cars. For our ridge regression model, we got an R^2 of 0.8512, an RMSE of 3709.2976, and an MAE of 2365.5228. This means that approximately 85% of the variability in the selling price can be explained from our independent variables. Once again, the high R^2 score indicates the model is effective at predicting car prices based on the provided features. Since our Ridge Regression model slightly outperforms the Linear Regression model, we can assume some features that were somewhat correlated with each other were penalized by introducing a regularization term in the ridge model. For our Lasso Regression model, we got an R^2 of 0.8284, an RMSE of 3983.6842, and an MAE of 2517.8088. This means that approximately 82% of the variability in the selling price can be explained from our independent variables. Since Lasso Regression shrinks the importance of some features, the slightly lower R^2 score may suggest that all the features we used in our model were important and not redundant.

![lin reg actual vs predicted](https://github.com/user-attachments/assets/824c0e74-7f18-4ead-aa55-2b9d180bc55c)

Figure 11. Scatterplot showing Actual Selling Price vs Predicted Selling Price

## Conclusion

In this project, we set out to predict the selling prices of used cars based on various car features using advanced regression techniques. Through a detailed process of data preprocessing, feature engineering, and model tuning, we built and evaluated several models, including Linear Regression, Ridge Regression, and Lasso Regression. Our goal was to identify the most accurate model and provide actionable insights for car dealerships, buyers, and sellers. After extensive data cleaning and feature selection, we trained and evaluated multiple regression models. Our models demonstrated strong performance, with the Ridge Regression model yielding the best results, achieving an R^2 of 0.8512, an RMSE of 3,709.3, and an MAE of 2,365.5. The Linear Regression model also performed well, with an R^2 of 0.8491, but the Ridge model slightly outperformed it, suggesting that regularization helped improve model accuracy by addressing multicollinearity. 

Ultimately, this analysis provides valuable insights into the factors influencing car prices and highlights the effectiveness of regression models in predicting these prices. The results of this study can assist car dealerships in pricing vehicles more competitively, help buyers make informed decisions, and support sellers in maximizing their returns. Moving forward, the integration of more advanced models such as Random Forest and XGBoost could further enhance predictive accuracy, especially in capturing nonlinear relationships in the data.

## Resources

https://www.kaggle.com/datasets/tunguz/used-car-auction-prices

<br />
<br />
<br />
<br />































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
