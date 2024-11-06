# CS506-Project Proposal

## Members: Ethan Machleder, Keith Yeung, Jason Zhu

There are many different factors that affect the price of a car, such as the brand, mileage, manufactoring year, and features of the car. For our project, we want to be able to successfully predict the price of cars based on these variables.

The data that needs to be collected is information on car sales and as many features of the cars as possible. We may be able to find this data through past car auctions, most likely from Kaggle datasets.

We plan on modeling the data by fitting a linear model to model the relationship between certain features and price. To model more complex relationships between multiple features and price, we can create decision trees or implement an XGBoost algorithm.

For basic descriptive visualizations, we will have bar graphs to compare different features as well as a correlation heatmap. We will also have interactive scatterplots so users can explore how different features impact car price.

We plan to split the collected data into training and testing sets, using 70-80% of the data for training and 20-30% for testing. To prevent overfitting and to validate our models, we will implement cross-validation techniques. Model performance will be evaluated using metrics such as Mean Squared Error (MSE).

Report:
In our preliminary data processing, we focused on identifying and handling missing values to ensure data integrity. Initially, we examined each column to quantify the number and percentage of missing values, giving us a clear overview of data completeness. Based on this assessment, we identified essential columns with significant gaps, including make, model, trim, body, odometer, color, interior, transmission, and condition. To retain data quality, we chose to drop rows where these key features were missing, as they represent critical information for our analysis. After this cleaning step, we confirmed the reduction in missing data by recalculating the number and percentage of null values, leading to a refined dataset that better supports accurate modeling and analysis.For each column, we calculated the frequency of unique values, providing a clear picture of the most and least represented categories. These counts offer a foundational understanding of our dataset's composition and can inform further analysis, such as identifying popular car makes or common body styles. We also standardized the text format in the body column by converting each entry to title case, ensuring consistency across values (e.g., converting "sedan" to "Sedan"). This step minimizes potential issues in downstream analyses caused by inconsistent text formats. Additionally, we saved the saledate column and the calculated frequency counts to text files to maintain a record of these key attributes for reference and ease of access in future analyses.To better understand our dataset, we employed various graphical visualizations that reveal trends, distributions, and relationships within key features. By plotting the frequency counts for categorical columns like make, model, trim, and body, we gained a clearer view of the distribution of car types, popular models, and common trim levels.
The box plots allowed us to examine the distribution and spread of selling prices within each category, revealing how specific attributes may impact vehicle pricing. Overall, these visualizations provide valuable insights into both numerical relationships and categorical trends, guiding us in selecting relevant features for further modeling.To evaluate our model’s performance, we made predictions on the test set (X_test) and compared them to the actual values (y_test). We calculated key evaluation metrics, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R-squared (R²) score. MSE provides the average squared difference between predicted and actual values, while RMSE, as the square root of MSE, indicates the model’s prediction error in the same units as the target variable. The R² score shows the proportion of variance in the selling price explained by our model, giving an overall measure of model fit. These metrics help us assess prediction accuracy and model effectiveness.








