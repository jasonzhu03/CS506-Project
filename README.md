# CS506-Project Proposal

## Members: Ethan Machleder, Keith Yeung, Jason Zhu

There are many different factors that affect the price of a car, such as the brand, mileage, manufactoring year, and features of the car. For our project, we want to be able to successfully predict the price of cars based on these variables, focusing on used cars from BMW.

The data that needs to be collected is information on car sales and as many features of the cars as possible. We may be able to find this data through past car auctions, most likely from Kaggle datasets.

We plan on modeling the data by fitting a linear model to model the relationship between certain features and price. To model more complex relationships between multiple features and price, we can create decision trees or implement an XGBoost algorithm.

For basic descriptive visualizations, we will have bar graphs to compare different features as well as a correlation heatmap. We will also have interactive scatterplots so users can explore how different features impact car price.

We plan to split the collected data into training and testing sets, using 70-80% of the data for training and 20-30% for testing. To prevent overfitting and to validate our models, we will implement cross-validation techniques. Model performance will be evaluated using metrics such as Mean Squared Error (MSE).

