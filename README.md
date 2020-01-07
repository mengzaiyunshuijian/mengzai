# Machine Learning Engineer Nanodegree
## Introduction and Foundations
## Project: Titanic Survival Exploration

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included

### Code

Template code is provided in the notebook `titanic_survival_exploration.ipynb` notebook file. Additional supporting code can be found in `visuals.py`. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` is meant to be used out-of-the-box and not intended for students to manipulate. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Run

In a terminal or command window, navigate to the top-level project directory `titanic_survival_exploration/` (that contains this README) and run one of the following commands:

```bash
jupyter notebook titanic_survival_exploration.ipynb
```
or
```bash
ipython notebook titanic_survival_exploration.ipynb
```

This will open the Jupyter Notebook software and project file in your web browser.

### Data

The dataset used in this project is included as `titanic_data.csv`. This dataset is provided by Udacity and contains the following attributes:

**Features**
- `pclass` : Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- `name` : Name
- `sex` : Sex
- `age` : Age
- `sibsp` : Number of Siblings/Spouses Aboard
- `parch` : Number of Parents/Children Aboard
- `ticket` : Ticket Number
- `fare` : Passenger Fare
- `cabin` : Cabin
- `embarked` : Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

**Target Variable**
- `survival` : Survival (0 = No; 1 = Yes)

### Analysis

In this forecast, we first analyzed the data to find that age, gender, class, and number of spouses, children, and brothers are important factors for survival, so we conducted an in-depth analysis to find the important factors that affect the survival rate:

1.Passengers younger than 10 survive.

2.Female passengers in 1st and 2nd class survive first. Females in 3rd class have a higher survival rate when the number of relatives is 0.

3.Survival rates for passengers aged 20 to 45 are higher
By analyzing the characteristics of the data, step-by-step debugging improves the prediction accuracy rate to 80.81%. At the same time, because there are only two output results in this case, we use a linear classifier to make the prediction. The result is 80.26%, which indicates that the decision tree The algorithm is more suitable for this case.

**Advantages** 

1.Through in-depth analysis of the data, reasonable division of branches can quickly improve the accuracy of predictions.It’s easy to understand and explain because the tree diagram can be drawn and seen.
2.Requires little data preparation. Many other algorithms usually require data normalization, creating dummy variables and deleting null values. Note, however, that the decision tree module in sklearn does not support the handling of missing values.

3.The cost of using a tree (for example, when predicting data) is the logarithm of the number of data points used to train the tree, which is a very low cost compared to other algorithms.

4.Ability to process both numeric and categorical data, both for regression and categorization. Other techniques are typically used to analyze data sets with only one variable type.

5.Able to deal with multiple output problems, that is, problems with multiple labels, pay attention to distinguishing from problems with multiple labels in one label.

6.It is a white box model and the results can be easily explained. If a given situation can be observed in the model, the conditions can be easily explained through Boolean logic. In contrast, in a black box model (for example, in an artificial neural network), the results may be more difficult to interpret.

**Disadvantages**

1.Only 6 branches of the data were divided, which resulted in a low accuracy, but met the requirements of the problem。Even if its assumption violates the real model of the generated data to some extent, it can perform well.

2.Decision tree learners may create overly complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required for leaf nodes or setting the maximum depth of the tree are necessary to avoid this problem, and the integration and adjustment of these parameters will be more obscure for beginners.

3.Decision trees may be unstable, and small changes in the data may lead to the generation of completely different trees. This problem needs to be solved by integrated algorithms.

## Conclusion

Through the guidance of Mr. Peter, I have a basic understanding of the theoretical framework of machine learning and the development status. After the teacher's in-depth analysis of code cases, I have a preliminary understanding of Python. After the class, I completed this assignment carefully and understood the principles of decision trees and the scope of their application, as well as the comparison between different algorithms, and laid a solid foundation for future machine learning.I will do more exercises in the future to better master the language of Python.