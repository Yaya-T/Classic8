
**About the models**

*Logistic Regression*: Logistic regression is a statistical model that in its basic form uses a logistic function to model the probability of a certain class

*LineasrSVC*: Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and scale better to large numbers of samples.This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.

*Multinomial Naive Bayes(MultiNB)*: Naive Bayes classifier for multinomial models. The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

*Support Vector Machine(SVC)*: The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.we have used the rbf kernel, which allows the SVC to fit a non-linear decision boundary.
