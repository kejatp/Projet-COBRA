class Machine:
  """
    Machine class for COBRA. method fit/predict and score can be improve to adapt better to the different machine that we use as an input for cobra. For now, it only work with sckit learn regressor.
  """
  def __init__(self,name,m,isTrain=False):
    self.name = name
    self.isTrain = isTrain
    self.m = m
  def fit(self,X_train,Y_train):
    if self.isTrain == False:
      self.isTrain=True
      return self.m.fit(X_train,Y_train)
    else:
      raise Exception(f"Machine {self.name} has already been trained")
  def predict(self,X):
    return self.m.predict(X)
  def score(self,X,Y):
    return self.m.score(X,Y)