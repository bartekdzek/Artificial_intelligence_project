# Machine Learning Project

This repository contains a project exploring the capabilities of various machine learning algorithms for two distinct problems: a regression problem (predicting diamond prices) and a classification problem (predicting weather conditions). All algorithms were implemented from scratch, adhering to the project's requirement of not using pre-built library implementations for the core methods.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Descriptions](#problem-descriptions)
  - [Regression Problem: Diamond Price Prediction](#regression-problem-diamond-price-prediction)
  - [Classification Problem: Weather Classification](#classification-problem-weather-classification)
- [Implemented Machine Learning Methods](#implemented-machine-learning-methods)
  - [Decision Tree (Drzewo Decyzyjne)](#decision-tree-drzewo-decyzyjne)
  - [K-Nearest Neighbors (K-Najbliższych Sąsiadów - KNN)](#k-nearest-neighbors-k-najbliższych-sąsiadów---knn)
  - [Stochastic Gradient Descent (SGD) Regression](#stochastic-gradient-descent-sgd-regression)
  - [Logistic Regression (Regresja Logistyczna)](#logistic-regression-regresja-logistyczna)
- [Parameter Analysis](#parameter-analysis)
  - [Regression Problem Analysis](#regression-problem-analysis)
  - [Classification Problem Analysis](#classification-problem-analysis)
- [Results and Conclusions](#results-and-conclusions)
- [How to Run](#how-to-run)
- [Contact](#contact)

## Project Overview

This project builds upon the problems explored in the Artificial Neural Networks project, applying various classical machine learning algorithms to solve them. The core objective was to implement at least three distinct machine learning methods for each problem (regression and classification) without relying on high-level libraries (like scikit-learn) for the algorithm's core logic. The project also involved a detailed analysis of how different parameters influence the effectiveness of each implemented method. For each parameter, at least four distinct values were tested.

## Problem Descriptions

### Regression Problem: Diamond Price Prediction

**Problem:** The regression task involves predicting the price of diamonds based on their characteristics.
**Dataset:** The project utilizes the `diamonds.csv` dataset, which includes detailed information on diamond parameters such as mass (carat), cut quality (cut), color (color), clarity (clarity), and physical dimensions.
**Goal:** To accurately forecast the continuous numerical `price` of a diamond given its attributes.

### Classification Problem: Weather Classification

**Problem:** The classification task is focused on categorizing weather conditions based on various meteorological data.
**Dataset:** The project uses a dataset containing meteorological features to classify weather into distinct categories (e.g., 'Sunny', 'Cloudy', 'Rainy', 'Snowy').
**Goal:** To assign a specific weather class to an observation based on input features.

## Implemented Machine Learning Methods

For this project, the following machine learning algorithms were implemented from scratch:

### Decision Tree (Drzewo Decyzyjne)

**Short Description:** A non-parametric supervised learning method used for both classification and regression. It works by creating a model of decisions based on actual feature values. It recursively partitions the data into subsets based on the most significant attribute at each node until a stopping criterion is met (e.g., maximum depth or minimum samples per leaf). The final model is a tree-like structure where internal nodes represent tests on attributes, branches represent outcomes of the test, and leaf nodes represent class labels (for classification) or predicted values (for regression).

### K-Nearest Neighbors (K-Najbliższych Sąsiadów - KNN)

**Short Description:** A non-parametric, instance-based learning algorithm. In KNN, the classification or regression of a new data point is based on the 'k' nearest training examples in the feature space. For classification, the output is a class membership (e.g., by majority vote of its neighbors). For regression, the output is the average of the values of its k nearest neighbors. The 'distance' between points is typically calculated using metrics like Euclidean or Manhattan distance.

### Stochastic Gradient Descent (SGD) Regression

**Short Description:** An iterative optimization algorithm used to minimize an objective function (cost function) for linear models. Instead of calculating the gradient over the entire dataset (as in Batch Gradient Descent), SGD updates the model's weights and bias based on the gradient of a *single* randomly chosen training example at each iteration. This makes it computationally much faster for large datasets, though its path to convergence can be more erratic due to the noisy updates.

### Logistic Regression (Regresja Logistyczna)

**Short Description:** Despite its name, Logistic Regression is a statistical model used for binary classification tasks (though it can be extended for multi-class classification). It models the probability of a certain class or event existing, typically using a logistic function (sigmoid function) to map predicted values to probabilities between 0 and 1. The model learns a linear relationship between input features and the log-odds of the outcome.

## Parameter Analysis

We analyzed the impact of various parameters on the performance of our implemented machine learning models. The analyses were performed separately for the regression and classification problems, with at least four different values tested for each significant parameter.

### Regression Problem Analysis (located in `regresja/`)

The `regresja.ipynb` notebook and its HTML export `regresja.html` contain the detailed analysis for the diamond price prediction problem. Data files and result Excel files are expected to be found within the `regresja/` directory.

**Key Findings (from `regresja.html`):**
-   **Decision Tree:** Achieved the best results among all models with `min_samples_split = 10.0`, yielding an impressive R² = 0.9756 on the test set.
-   **SGD Regression:** Optimal results were obtained with `learning_rate = 0.001` and `batch_size = 128`, leading to R² = 0.9007 on test data.
-   **K-Nearest Neighbors (KNN):** Showed the highest effectiveness with `10 neighbors` and the `Manhattan metric (p=1)`, achieving R² = 0.957 during testing.
-   **Overall:** The Decision Tree proved to be the most effective model for this regression task, demonstrating the highest coefficient of determination (R²) and the lowest RMSE and MAE errors. All models were implemented from scratch, ensuring full control over the learning process and parameter selection.

### Classification Problem Analysis (located in `klasyfikacja/`)

The `klasyfikacja.ipynb` notebook and its HTML export `klasyfikacja.html` contain the detailed analysis for the weather classification problem. Data files and result Excel files are expected to be found within the `klasyfikacja/` directory.

**Key Findings (from `klasyfikacja.html`):**
-   **Decision Tree:** Achieved the best results with parameters `max_depth=8` and `min_samples_split=100`, resulting in high classification accuracy.
-   **Logistic Regression:** Demonstrated the best effectiveness with a `learning_rate=0.2` and `2000 iterations`, allowing for stable model training.
-   **K-Nearest Neighbors (KNN):** Achieved optimal results using the `Manhattan metric` and a `min_label strategy` for resolving ties.
-   **Overall:** All three implemented models achieved an accuracy level of approximately 95-96%, indicating their high effectiveness in the task of classifying weather conditions. Various metrics such as accuracy, precision, recall, and F1-score were used to evaluate model performance, confirming the high quality of classification.
-   Data preprocessing involved encoding categorical variables and removing outliers using the IQR method.

## Results and Conclusions

This project successfully demonstrates the implementation and evaluation of fundamental machine learning algorithms from scratch. Key takeaways include:

-   The performance of classical machine learning models is highly dependent on careful parameter tuning.
-   Even without relying on advanced libraries, well-implemented basic algorithms can achieve high accuracy on complex problems.
-   For the regression problem (diamond prices), the Decision Tree proved to be exceptionally effective.
-   For the classification problem (weather conditions), all three implemented methods (Decision Tree, Logistic Regression, KNN) performed strongly, with slight variations depending on the metric and parameter configurations.
-   The "from-scratch" implementation provided a deeper understanding of the internal workings of each algorithm.
