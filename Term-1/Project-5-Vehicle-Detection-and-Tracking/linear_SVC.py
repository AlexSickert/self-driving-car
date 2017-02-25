from sklearn.svm import LinearSVC


# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Train the SVC
svc.fit(X_train, y_train)

#Then you can check the accuracy of your classifier on the test dataset like this:

print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

#Or you can make predictions on a subset of the test data and compare directly with ground truth:

print('My SVC predicts: ', svc.predict(X_test[0:10].reshape(1, -1)))
print('For labels: ', y_test[0:10])

#Play with the parameter values spatial and histbin in the exercise below to see how the classifier accuracy and training time vary with the feature vector input.

#==============================================================================
