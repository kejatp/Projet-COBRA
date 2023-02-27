class Dataset:
  """
    Dataset class for COBRA
  """
  def __init__(self,X_train,Y_train,X_test,Y_test):
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  def split(self,ratio):
      # ratio = 0.5 si on veut séprarer en taille égale (si paire).
      k, = np.shape(self.Y_train)
      self.l = int(k*ratio)
      self.X_train_BM,self.X_train_cobra = self.X_train[0:self.l],self.X_train[self.l:]
      self.Y_train_BM,self.Y_train_cobra = self.Y_train[0:self.l],self.Y_train[self.l:]