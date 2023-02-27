import dataset
import machine
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model

class COBRA:
    """
      Main class of this project. Implement all the methods necessary to train and evaluate COBRA.

      Attribut:
        - nb_machines [int] : number of machine in COBRA
        - machines [dictionary] : dictionary of all machine in COBRA. self.machines["name"] = machine where machine is an object of Machine() class define before.
        - ratio [float] : ratio choosen for splitting the training set between D_l and D_k.
        - isCobraTrain [bool] : boolean that make sure COBRA is train before evaluating it.
        - score_training [dictionary] : dictionary that keep tracks of the performance of the different machines on the training set.
        - score_test [dictionary] : dictionary that keep tracks of the performance of the different machines on the test set.
        - epsilon [float] : the threshold that we see on the difinition of the COBRA estimator in equation (1)
        - preds [dictionary] : dictionary that keep track of the prediction made by the machines for a new dataset X.
        - ds [Dataset()] : the dataset that we use to evaluate and train COBRA   

      Methods:
        - __init__(self,ratio=0.5): Initialisation of the COBRA regressor.
        - setEpsilon(self,new_epsilon) : change epsilon to the value of new_epsilon
        - chooseEpsilon(self,X,Y) : choose epsilon for a specific dataset according to equation (2)
        - addMachine(self,name,m) : add a new machine to the dictionnary machines.
        - trainMachines(self,name='ALL') : train the machine "name". If name == 'ALL', train all the machine that are not yet trained.
        - pretrain(self,verbose=False) : pretrain COBRA by calculating the prediction of all the machines on the dataset D_l.
        - predict(self,X,verbose=False) : make prediction on the dataset X.
        - evaluate(self,metrics='R square') : evaluate the COBRA regressor on the test set. 

      yet to be implemented:
        - self.alpha [float] : parameters between 0 and 1 that tell us the proportion of the unanimity.
      
    """
    def __init__(self,ratio=0.5):
        self.nb_machines = 0
        self.machines = {}
        self.ratio = ratio
        self.isCobraTrain = False
        self.scores_training = {}
        self.scores_test = {}
        self.epsilon = 0
        self.preds = {}
        self.alpha = 1

    def setEpsilon(self,new_epsilon):
      self.epsilon = new_epsilon
    
    def setAlpha(self,new_alpha):
      self.alpha = new_alpha

    def chooseEpsilon(self,X,Y):
      max = - np.Infinity
      min = np.Infinity
      for name, machine in self.machines.items():
        temp_max = np.max(machine.predict(X))
        temp_min = np.min(machine.predict(X))
        if temp_max > max:
          max = temp_max
        if temp_min < min:
          min = temp_min
      self.epsilon = max - min 
    
    def hyperparamOpti(self,X,Y):
      max = - np.Infinity
      min = np.Infinity
      for name, machine in self.machines.items():
        temp_max = np.max(machine.predict(X))
        temp_min = np.min(machine.predict(X))
        if temp_max > max:
          max = temp_max
        if temp_min < min:
          min = temp_min
      l_alpha = [ 1/i for i in range(1,self.nb_machines+1) ]
      l_epsilon = np.linspace(10**(-300),max-min,20)
      best_epsilon = 0
      best_alpha = 0
      score = 0
      for i,alpha in enumerate(l_alpha):
        self.alpha = alpha
        for j,epsilon in enumerate(l_epsilon):
          print(f"Iteration number {i*len(l_epsilon)+j}/{len(l_alpha)*len(l_epsilon)} : \t epsilon = {self.epsilon} \t alpha = {self.alpha}")
          self.epsilon = epsilon
          try :
            precision,new_score = self.evaluate(X,Y,all=True)
            print(f'loss is {new_score}')
            print(f'precision is {precision}')
            if self.evaluate(X,Y,metrics='quadratic loss') > score:
              best_epsilon = epsilon
              best_alpha = alpha
          except (Exception):
            pass
      self.epsilon = best_epsilon
      self.alpha = best_alpha

    def addMachine(self,name,m):
      newM = Machine(name,m)
      self.machines[newM.name] = newM
      self.nb_machines+=1

    def addDataset(self,X_train,Y_train,X_test,Y_test):
      self.ds = Dataset(X_train,Y_train,X_test,Y_test)
      self.ds.split(self.ratio)

    def trainMachines(self,name='ALL'):
      if name=='ALL':
        for name,machine in self.machines.items():
          if machine.isTrain ==False:
            print(f"Training of machine {name}")
            machine.fit(self.ds.X_train_BM,self.ds.Y_train_BM)
            score_training = machine.score(self.ds.X_train_BM,self.ds.Y_train_BM)
            score_test = machine.score(self.ds.X_test,self.ds.Y_test)
            self.scores_training[name] = score_training
            self.scores_test[name] = score_test
            print(f"Machine {name} score on training set is:{score_training}")
            print(f"Machine {name} score on test set is:{score_test}")
            print("\n===================================================")

      else:
        if self.machines[name].isTrain == False:
          self.machines[name].fit(self.ds.X_train_BM,self.ds.Y_train_BM)
          self.machines[name] = machine
          score_training = machine.score(self.ds.X_train_BM,self.ds.Y_train_BM)
          score_test = machine.score(self.ds.X_test,self.ds.Y_test)
          self.scores_training[name] = score_training
          self.scores_test[name] = score_test
          print(f"Machine {name} score on training set is:{score_training}")
          print(f"Machine {name} score on test set is:{score_test}")
          print("\n===================================================")

        else:
          raise Exception("Machine déjà entrainer")

    def pretrain(self,verbose=False):
      for name,machine in self.machines.items():
        if machine.isTrain == False:
          raise Exception(f"Machine {name} is not train yet, you have to train it first")
      if verbose:
        print("All the machine have been train, we can now pretrain COBRA")
      if self.isCobraTrain:
        raise Exception("COBRA is already pretrain")
      for name,machine in self.machines.items():
        if verbose:
          print(f"Making prediction for machine {name}")
        self.preds[name] = machine.predict(self.ds.X_train_cobra)
      self.isCobraTrain = True

    def predict(self,X,verbose=False):
      if self.epsilon == 0:
        raise Exception ("Epsilon has not yet been choose")
      if self.isCobraTrain == False:
        raise Exception ("Cobra is not pretrain yet, please execute the method .pretrain_corbra() first")
      if verbose:
        print("COBRA is pretrain, we can now enter in the prediction phase")
      pred_x = {}
      res = np.ones(len(X))
      if verbose:
        print(f'size of the test set is : {len(X)}')
      size_intervalle = len(self.ds.X_train_cobra) //100
      res={}
      for name,machine in self.machines.items():
        if verbose:
          print(f"Prediction for {name}")
        cpt = 0
        pred = machine.predict(X)
        res_machine = []
        for i in range(len(self.ds.X_train_cobra)):
          if verbose: #useless au final
            if i % size_intervalle == 0:
              cpt+=1
              print("[" + "=" *cpt + " "*(100-cpt) + "]" + f"    example number = {i}/{len(self.ds.X_train_cobra)}")
          temp_array = np.ones(len(pred))*self.preds[name][i]
          #print(f"content of the res_machine.append{np.abs(pred - temp_array)}")
          #print(f"temp array {temp_array}")
          #print(f"np.where result is {np.where(np.abs(pred - temp_array) <= self.epsilon,1,0)}")
          res_machine.append(np.where(np.abs(pred - temp_array) <= self.epsilon,1,0))
        res[name]= np.array(res_machine)
      #print(f"res_machine is {res_machine}")
      final_res = np.ones((len(self.ds.X_train_cobra),len(X),self.nb_machines)) #modify by adding self.nb_machines
      for i,keys in enumerate(res):
        final_res[:,:,i] = res[keys]
      final_res = np.sum(final_res,axis=-1) #We sum on the number of machines.
      #print(final_res.shape)
      if verbose:
        print(f"Number of non zeros value is {np.count_nonzero(final_res)}")
        print(f"Number of 0 is {np.count_nonzero(final_res == 0)}")
      indices=[]
      normalisation_factor = []
      y_pred = np.zeros(len(X))
      #print(f"final res is: {final_res}")
      for j in range(len(final_res[0])):
        #print(f"np where result {np.where(final_res[:,j] >= self.alpha * self.nb_machines )}")
        indices.append(np.where(final_res[:,j] >= self.alpha * self.nb_machines )[0])
        #normalisation_factor.append(np.sum(final_res[:,j]))
        normalisation_factor.append(np.sum(np.where(final_res[:,j] >= self.alpha * self.nb_machines,0,1)))
        #print(np.sum(final_res[:,j]))
      #print(normalisation_factor[2])
      ##print(indices[0])
      for i in range(len(indices)):
        if normalisation_factor[i] ==0:
          raise Exception (f"Can't predict example {i} in the training set because no label is close enought")
        sum=0
        for indice in indices[i]:
          #print(indice)
          #print(self.ds.Y_train_cobra.iloc[indice])
          sum+= self.ds.Y_train_cobra.to_numpy()[indice]
        sum=sum/normalisation_factor[i]
        y_pred[i]=sum
      return(y_pred)

    def evaluate(self,X,y_true,all=True,metrics='R square'):
      if all:
        y_pred = self.predict(X,verbose=False)
        n = len(y_true)
        u = np.sum(np.power((y_true - y_pred),2))
        v = np.sum(np.power((y_true - np.mean(y_true)),2))
        precision = 1 - (u/v)
        loss = (1/n) * (np.sum(np.power(y_pred - y_true,2)))
        return precision,loss
      if metrics == 'R square':
        y_pred = self.predict(X,verbose=False)
        #print(type(y_pred))
        #print(y_pred)
        #print(type(y_true))
        #print(y_true)
        u = np.sum(np.power((y_true - y_pred),2))
        v = np.sum(np.power((y_true - np.mean(y_true)),2))
        return(1-u/v)
      if metrics.lower() == 'quadratic loss':
        y_pred = self.predict(X,verbose=False)
        n = len(y_true)
        return (1/n) * (np.sum(np.power(y_pred - y_true,2)))

      else:
        raise Exception("Metrics non implemented yet")
