# Learning Machine Learning Algorithms

This is my repository for the the things that I have learned about implementing machine learning algorithms from scratch or using libraries.

## [univariate_linear_regression.ipynb](https://github.com/ignadr/ML-Algorithms-Manual/blob/main/univariate_linear_regression.ipynb)<br />
Learned about univariate linear regression, computing the cost using mean squared error, and applying gradient descent to find the optimal parameters that minimize the cost. The dataset used is a mock dataset using numpy array.
<br />
<br />
## [experience_vs_salary.ipynb](https://github.com/ignadr/ML-Algorithms-Manual/blob/main/experience_vs_salary.ipynb)<br />
Implementing the cost function and gradient descent for univariate linear regression to a dataset from [Kaggle](https://www.kaggle.com/).<br />
Link to dataset: [Salary](https://www.kaggle.com/datasets/rsadiq/salary)
<br />
<br />
## [multiple_linear_regression.ipynb](https://github.com/ignadr/ML-Algorithms-Manual/blob/main/multiple_linear_regression.ipynb)<br />
Learned about multiple linear regression, computing the cost using mean squared error, and applying gradient descent to find the optimal parameters that minimize the cost. The dataset used is a mock dataset using numpy array.
<br />
<br />
## [auto_mpg.ipynb](https://github.com/ignadr/ML-Algorithms-Manual/blob/main/auto_mpg.ipynb)
Implementing the cost function and gradient descent for multiple linear regression to a dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg).<br />
<br />
Initially, I implemented univariate linear regression using horsepower as feature and mpg (miles per gallon) as the target. However, the dataset has multiple features other than horsepower. So, I decided I will use it for multiple linear regression.<br />
Link to dataset: [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)
<br />
<br />
## [student_performance_multiple_LR.ipynb](https://github.com/ignadr/ML-Algorithms-Manual/blob/main/student_performance_multiple_LR.ipynb)
Still on the topic of multiple linear regression, I want to try it with another dataset. However, I also try implementing z-score normalization and splitting the dataset into training and testing set.<br />
Link to dataset: [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
<br />
<br />
## [mushroom_binary_classification.ipynb](https://github.com/ignadr/ML-Algorithms-Practice/blob/main/mushroom_binary_classification.ipynb)
I started learning binary classification using logistic regression, this time using sklearn. This is also when I refresh my knowledge about `OneHotEncoder` from sklearn.<br />
Link to dataset: [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
<br />
<br />
## [drug_classification.ipynb](https://github.com/ignadr/ML-Algorithms-Practice/blob/main/drug_classification.ipynb)
Still learning about classification, but this one is multiclass classification. Overall, I learned a lot about transforming multiple features at the same time using `ColumnTransformer` from sklearn and also using `OrdinalEncoder` to transform ordinal data and `OneHotEncoder` to transform nominal data. During this time, I also learned about ANN using tensorflow.<br />
Link to dataset: [Drug Classification](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)
<br />
<br />
## [titanic_survival_prediction.ipynb](https://github.com/ignadr/ML-Algorithms-Practice/blob/main/titanic_survival_prediction.ipynb)
I tried joining the popular [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition in Kaggle. I learned a lot about the importance of data cleaning and feature engineering. A clean dataset goes a long way. Although my feature engineering is still lacking for now, I am satisfied that I managed to join this competition. There are some ideas of which features I want to transform and create next, so I will probably come back to this dataset later in the future.<br />
Latest score: 0.78468<br />
Link to my model in Kaggle: [Titanic Survival Prediction](https://www.kaggle.com/code/ignatiusadrian/titanic-survival-prediction)
<br />
<br />
## [wine_clustering_k_means_practice.ipynb](https://github.com/ignadr/ML-Algorithms-Practice/blob/main/wine_clustering_k_means_practice.ipynb)
Implemented K-Means algorithm using sklearn to a wine dataset. It is a very basic implementation of K-Means to a basic dataset, but what I learned the most is that I can visualize a dataset with more than 2 features if I applied PCA first, which I think is interesting.<br />
Link to dataset: [Wine Dataset for Clustering](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering/data)
<br />
<br />
## [book_recommender.ipynb](https://github.com/ignadr/ML-Algorithms-Practice/blob/main/book_recommender.ipynb)
Learning about recommender system with `cosine_similarity` from sklearn. For most of the codes, I learned a lot from this [notebook](https://www.kaggle.com/code/shivamja/books-recommendation-system). It kind of gave me a sense on how I should approach the next recommender system practice (when I actually practice it again).<br />
Link to dataset: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
