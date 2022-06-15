from pyexpat import features
import random
import math
import os
from turtle import shape
from xml.sax.handler import property_interning_dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from sklearn import linear_model, svm, discriminant_analysis, metrics
from scipy import optimize
import seaborn as sns
# Provides the methods to create and solve the firefighter problem
class FFP:

  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")    
    text = file.read()    
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))  
    self.result = 0
    self.featuresActualState = []
    self.trainingFeatures = [] #X, features results
    self.trainingClass = [] #Y, heuristic
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1      
    self.graph = []    
    for i in range(self.n):
      self.graph.append([0] * self.n);
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1    

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))    
    t = 0
    while (spreading):
      #always in order/
      # EDGE_DENSITY AVG_DEGREE BURNING_NODES BURNING_EDGES NODES_IN_DANGER
      featuresList = []
      featuresList.append(self.getFeature("EDGE_DENSITY"));
      featuresList.append(self.getFeature("AVG_DEGREE"));
      featuresList.append(self.getFeature("BURNING_NODES"));
      featuresList.append(self.getFeature("BURNING_EDGES"));
      featuresList.append(self.getFeature("NODES_IN_DANGER"));
      self.featuresActualState = featuresList
      self.trainingFeatures.append(featuresList)
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        heuristic = method
        if (isinstance(method, HyperHeuristic)):
          heuristic = method.nextHeuristic(self)
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected   
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))            
      # It spreads the fire among the unprotected nodes
      spreading = False 
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1): 
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))     
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):    
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    print("how many steps", len(self.trainingFeatures))
    self.trainingClass = [method]*len(self.trainingFeatures)
    self.result = self.getFeature("BURNING_NODES")
    
  def retrieveFeaturesActualState(self):
    return self.featuresActualState

  def whatIsResult(self):
    return self.result

  def transformAndPrintTrainingX(self, flag = False):
    self.trainingFeatures = np.array(self.trainingFeatures)
    if(flag):
      print(self.trainingFeatures)
  
  def transformAndPrintTrainingY(self, flag = False):
    self.trainingClass = np.array(self.trainingClass)
    if(flag):
      print(self.trainingClass)
  
  def obtainTrainingData(self):
    return self.trainingFeatures, self.trainingClass    

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def __nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i        
        break
    value = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])              
              break
        elif (heuristic == "GDEG"):        
          value = sum(self.graph[i])          
        else:
          print("=====================")
          print("Critical error at FFP.__nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")          
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)      
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph) 
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph) 
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))    
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:      
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")          
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text

class SVM:
  def __init__(self):
    print("initializing SVM")

    self.colors = ['blue','red']
    self.cmap = pltcolors.ListedColormap(self.colors)
    self.nFeatures = 2
    self.N = 100
    self.model = svm.SVC(kernel='rbf', C = 10, gamma = 1/2, shrinking = False)

  def train(self, X, Y):
    XT = np.array(X)
    YT = np.array(Y)
    print("inside")
    print(XT.shape)
    print(YT.shape)
    self.model.fit(XT,YT)

  def predict(self, featuresToPredict):
    return self.model.predict(featuresToPredict)
  
  def plotLine(self, ax, xRange, w, x0, label, color='grey', linestyle='-', alpha=1.):
    """ Plot a (separating) line given the normal vector (weights) and point of intercept """
    if type(x0) == int or type(x0) == float or type(x0) == np.float64:
        x0 = [0, -x0 / w[1]]
    yy = -(w[0] / w[1]) * (xRange - x0[0]) + x0[1]
    ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)

  def plotSVM(self, X, y, support=None, w=None, intercept=0., label='Data', separatorLabel='Separator', ax=None, bound=[[0., 1.], [0., 1.]]):
    """ Plot the SVM separation, and margin """
    if ax is None:
        fig, ax = plt.subplots(1)
    
    im = ax.scatter(X[:,0], X[:,1], c=y, cmap=self.cmap, alpha=0.5, label=label)
    if support is not None:
        ax.scatter(support[:,0], support[:,1], label='Support', s=80, facecolors='none', 
                   edgecolors='y', color='y')
        print("Number of support vectors = %d" % (len(support)))
    if w is not None:
        xx = np.array(bound[0])
        self.plotLine(ax, xx, w, intercept, separatorLabel)
        # Plot margin
        if support is not None:
            signedDist = np.matmul(support, w)
            margin = np.max(signedDist) - np.min(signedDist) * np.sqrt(np.dot(w, w))
            supportMaxNeg = support[np.argmin(signedDist)]
            self.plotLine(ax, xx, w, supportMaxNeg, 'Margin -', linestyle='-.', alpha=0.8)
            supportMaxPos = support[np.argmax(signedDist)]
            self.plotLine(ax, xx, w, supportMaxPos, 'Margin +', linestyle='--', alpha=0.8)
            ax.set_title('Margin = %.3f' % (margin))
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])
    cb = plt.colorbar(im, ax=ax)
    loc = np.arange(-1,1,1)
    cb.set_ticks(loc)
    cb.set_ticklabels(['-1','1'])

  

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
class HyperHeuristic:

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  def __init__(self, features, heuristics):
    if (features):
      self.features = features.copy()
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of features cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
    if (heuristics):
      self.heuristics = heuristics.copy()
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of heuristics cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    print("=====================")
    print("Critical error at HyperHeuristic.nextHeuristic.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

  # Returns the string representation of this hyper-heuristic 
  def __str__(self):
    print("=====================")
    print("Critical error at HyperHeuristic.__str__.")
    print("The method has not been overriden by a valid subclass.")
    print("The system will halt.")
    print("=====================")
    exit(0)

class PabloHH(HyperHeuristic):
  def __init__(self, features, heuristics, model):
    super().__init__(features, heuristics)
    self.model = model

  def nextHeuristic(self, problem):
    featuresInStep = np.array(problem.retrieveFeaturesActualState())
    heuristic = self.model.predict([featuresInStep])
    print(featuresInStep)
    print("\t\t=> " + str(heuristic) )
    return heuristic

  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics)
    return text

# Tests
# =====================

isTraining = True
trainingDirectory = 'instances/onlytests'
testingDirectory = 'instances/testing'
# iterate over files in
# that trainingDirectory


trainingX = []
trainingY = []

if(isTraining):
  for filename in os.listdir(trainingDirectory):
      f = os.path.join(trainingDirectory, filename)
      # checking if it is a file
      if os.path.isfile(f):
        print(f)
        
        # Solves the problem using heuristic LDEG and one firefighter
        problem1 = FFP(f)
        problem1.solve("LDEG", 1, False)
        print("LDEG = " + str(problem1.whatIsResult()))

        # Solves the problem using heuristic GDEG and one firefighter
        problem2 = FFP(f)
        problem2.solve("GDEG", 1, False)
        print("GDEG = " + str(problem2.whatIsResult()))

        trainingXAux = []
        trainingYAux = []
        if(isTraining):
          if(problem1.whatIsResult() <= problem2.whatIsResult()):
            problem1.transformAndPrintTrainingX(0)
            problem1.transformAndPrintTrainingY(0)
            trainingXAux, trainingYAux = problem1.obtainTrainingData()
          else:
            problem2.transformAndPrintTrainingX(0)
            problem2.transformAndPrintTrainingY(0)
            trainingXAux, trainingYAux = problem2.obtainTrainingData()

          trainingX.extend(trainingXAux)
          trainingY.extend(trainingYAux)
          print(trainingX)
          print(trainingY)

supportVectorMachine = SVM()

if(isTraining):

  #train
  supportVectorMachine.train(trainingX, trainingY)

  #dump into Pickle
  print("trained")
else:
  problem3 = FFP(f)
  hh = PabloHH(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER", "BURNING_EDGES", "AVG_DEGREE"], ["LDEG", "GDEG"], supportVectorMachine)
  print(hh)
  problem3 = FFP(f)
  problem3.solve(hh,1,False)

  print("PabloHH = " + str(problem3.whatIsResult()))
  print()




