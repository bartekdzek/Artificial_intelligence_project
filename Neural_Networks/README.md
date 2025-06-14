# Artificial Neural Networks Project

This repository contains a project exploring the capabilities of Artificial Neural Networks (ANNs) for two distinct problems: one regression problem and one classification problem. The goal was to implement ANNs from scratch and analyze the impact of various parameters on their performance.


## Table of Contents


- [Project Overview](#project-overview)

- [Problem Descriptions](#problem-descriptions)

  - [Regression Problem](#regression-problem)

  - [Classification Problem](#classification-problem)

- [Literature Review](#literature-review)

- [Parameter Analysis](#parameter-analysis)

  - [Common Parameters Tested](#common-parameters-tested)

  - [Regression Problem Analysis](#regression-problem-analysis)

  - [Classification Problem Analysis](#classification-problem-analysis)

- [Results and Conclusions](#results-and-conclusions)


## Project Overview


The project focuses on building and evaluating Artificial Neural Networks for predicting numerical values (regression) and categorizing observations into predefined classes (classification). A key aspect of this project was to implement the neural networks without relying on high-level, pre-built libraries for network creation and training, allowing for a deeper understanding of their underlying mechanisms.


We investigated the influence of several key parameters on network effectiveness, including:

- Size of training and testing sets

- Type of activation function

- Number of neurons in hidden layers

- Learning method (optimizer)


For each parameter, at least four different values were tested, and the results for both training and testing sets are presented. Due to the non-deterministic nature of network training, each configuration was run multiple times, and the presented results reflect average or best performance metrics.


## Problem Descriptions

### Regression Problem

**Problem:** This section addresses a regression task where the goal is to predict a continuous numerical value. The dataset and specific target variable are detailed in the `regresjaNN.ipynb` notebook.

### Classification Problem

**Problem:** This section focuses on a classification task aimed at categorizing observations into distinct classes. The dataset and specific target classes are described in the `klasyfikacjaNN.ipynb` notebook. 

## Literature Review

This section provides an overview of relevant literature concerning the tackled problems and general concepts in Artificial Neural Networks.

**General ANNs and Learning Methods:**

- [https://prac.im.pwr.edu.pl/~zak/Pawel_Gburzynski_Sztuczna_inteligencja_cz_3_Sieci_neuronowe.pdf](https://prac.im.pwr.edu.pl/~zak/Pawel_Gburzynski_Sztuczna_inteligencja_cz_3_Sieci_neuronowe.pdf)

- [https://gdudek.el.pcz.pl/files/SI/SI_wyklad6.pdf](https://gdudek.el.pcz.pl/files/SI/SI_wyklad6.pdf)

- [https://media.statsoft.pl/_old_dnn/downloads/sieci%20neuronowe.pdf](https://media.statsoft.pl/_old_dnn/downloads/sieci%20neuronowe.pdf)

- [https://course.elementsofai.com/pl/5/2](https://course.elementsofai.com/pl/5/2)


## Parameter Analysis

We analyzed the impact of various parameters on the performance of our neural networks. The results are presented separately for the regression and classification problems.

### Common Parameters Tested


For both problems, we investigated the influence of:

- **Rozmiar zbioru testowego i treningowego** (Size of test and training sets)

- **Rodzaj funkcji aktywacji** (Type of activation function)

- **Liczba neuronów** (Number of neurons in hidden layers)

- **Metoda uczenia** (Learning method/Optimizer)


### Regression Problem Analysis


The `regresjaNN.ipynb` notebook and its HTML export `regresjaNN.html` contain the detailed analysis for the regression problem.


**Key Findings (from `regresjaNN.html`):**

- **Batch Gradient Descent (Batch GD)** consistently showed the highest RMSE, MAE, and MAPE values for both training and testing sets, indicating the weakest predictive performance.

- **Mini-Batch Gradient Descent (Mini-Batch GD), Adam Optimizer, and RMSProp** achieved very high R2 values for both training and testing sets, demonstrating their high effectiveness and good model generalization.

- **Batch GD** performed the worst in every tested scenario.

- **Mini-Batch GD** showed dominance across all tests, suggesting it is the most versatile and effective learning method among those compared.


### Classification Problem Analysis


The `klasyfikacjaNN.ipynb` notebook and its HTML export `klasyfikacjaNN.html` contain the detailed analysis for the classification problem.



**Key Findings (from `klasyfikacjaNN.html`):**

- **Momentum** achieved the highest accuracy on the training set, while **Mini-Batch Gradient Descent** achieved the highest accuracy on the test set.

- The **Momentum** function provided the highest values for all three metrics (precision, recall, and F1) for the "Sunny" class.

- Most classes, with the exception of "Snowy," achieved their highest F1 scores using the **Momentum** method.

- **Batch Gradient Descent** performed the worst across all examined aspects.

## Results and Conclusions


Based on our experiments, several key conclusions can be drawn:


**General:**

- The choice of learning method significantly impacts network performance, with Mini-Batch Gradient Descent and Momentum often outperforming Batch Gradient Descent.

- Thorough parameter tuning is crucial for optimizing ANN performance on specific tasks.

- The non-deterministic nature of network training necessitates multiple runs to obtain reliable average or best-case results.


**Regression Specific:**

- Mini-Batch GD, Adam, and RMSProp are effective optimizers for regression tasks, showing strong generalization capabilities.

- Batch GD is generally inefficient and prone to higher errors.

**Classification Specific:**

- Momentum and Mini-Batch Gradient Descent demonstrate strong performance in classification, with varying strengths depending on the metric and dataset.

- Batch Gradient Descent consistently underperforms in classification tasks as well.

The project highlights the importance of selecting appropriate network architectures, activation functions, and, most critically, learning algorithms for achieving optimal results in both regression and classification problems using custom-implemented neural networks.
