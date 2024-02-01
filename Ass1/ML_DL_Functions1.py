import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 315333997
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  Xt = np.hstack([X, np.ones((X.shape[0], 1))])
  theta = np.dot(np.matmul(np.linalg.inv(np.matmul(Xt.T, Xt)), Xt.T), y)
  return theta

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  pred_s = model.predict(X)
  return 100 * (np.sum(pred_s == s) / len(s))

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-5.60104810e-02, 3.73959200e-02, -3.33727853e-02,  3.95450109e-02,
 -3.08215127e-02, -2.41596486e-03, 7.43219163e-02,  1.58064254e-02,
  1.62477934e-02, -4.77338730e-02, 4.97174216e-02,  2.23789046e-02,
  9.69579270e-02, 1.62581810e-01, 7.53441574e-01,  2.35805279e-02,
  2.15328986e-02, 1.77235285e-02, 4.71508462e-03, -4.05292159e-04,
  3.06129660e-02, 2.41223836e-02, 1.36845964e-02, -3.43531339e-02,
 -3.03696739e-02, 4.01066341e-02, -2.82291201e-02, -3.73894721e-02]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return -1.2759414611776407e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-2.32346721e-01, -2.26330016e-01,  3.73492277e-01,  2.89845081e-04,
  -8.19131150e-02, -4.93346095e-01, -3.29547168e-03, -4.17147046e-02,
   5.45240567e-02,  2.66233029e-01, -2.56349719e-01, -7.05752413e-02,
  -3.87044703e-02,  7.82704634e-01,  3.02500315e+00, -5.88712731e-01,
   7.03974186e-02, -4.52925538e-02, -2.39991645e-02, -7.60048344e-02,
  -3.35655017e-02,  6.25422731e-02,  8.70829377e-02, -3.75129173e-02,
  -3.98792382e-01,  8.28929228e-02,  5.83071248e-02,  4.01259882e-01]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.28958071]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]