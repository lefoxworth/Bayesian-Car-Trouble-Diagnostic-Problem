#CS465 Homework #1 - Bayesian Network Inference
#Created by: Laura Foxworth
#Based on the probability.py code in the aima-python code base

from collections import defaultdict, Counter
import itertools
import math
import random

class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."

    def __init__(self):
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs

    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self

class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."

    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT

    def __repr__(self): return self.__name__

class Factor(dict): "An {outcome: frequency} mapping."

class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping.
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)

class Evidence(dict):
    "A {variable: value} mapping, describing what we know for sure."

class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."

    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table.
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple):
                row = (row,)
            self[row] = ProbDist(dist)

class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'

T = Bool(True)
F = Bool(False)

def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]

def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist

def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random() # r is a random point in the probability distribution
    c = 0.0             # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome

def globalize(mapping):
    "Given a {name: value} mapping, export all the names to the `globals()` namespace."
    globals().update(mapping)

def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})

def all_rows(net): return itertools.product(*[var.domain for var in net.variables])

def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]

def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result

def enumeration_ask(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i    = net.variables.index(X) # The index of the query variable X in the row
    dist = defaultdict(float)     # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)

def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)

##############################

probs = []

with open('data', 'r') as f:
    for line in f:
        probs.append(line.split())

#print(probs)

car_net = (BayesNet()
    .add('IW', [], float(probs[0][0]))
    .add('B', ['IW'], {T: float(probs[1][0]), F: float(probs[1][1])})
    .add('SM', ['IW'], {T: float(probs[2][0]), F: float(probs[2][1])})
    .add('R', ['B'], {T: float(probs[3][0]), F: float(probs[3][1])})
    .add('I', ['B'], {T: float(probs[4][0]), F: float(probs[4][1])})
    .add('G', [], float(probs[5][0]))
    .add('S', ['I','SM','G'], {(T, T, T): float(probs[6][0]), (T, T, F): float(probs[6][1]), (T, F, T): float(probs[6][2]), (T, F, F): float(probs[6][3]), (F, T, T): float(probs[6][4]), (F, T, F): float(probs[6][5]), (F, F, T): float(probs[6][6]), (F, F, F): float(probs[6][7])})
    .add('M', ['S'], {T: float(probs[7][0]), F: float(probs[7][1])}))

globalize(car_net.lookup)
#print(car_net.variables)

#print(joint_distribution(car_net))

while(1):

    print('Example query: IW=true,B=false,R=false')
    print('Accepted abbreviations:')
    print('IW=Icy Weather  B=Battery  SM=StarterMotor  R=Radio')
    print('I=Ignition  G=Gas  S=Starts  M=Moves\n')
    print('Type "exit" to quit the program\n')

    query = input('Enter your query hypothesis: ')
    #print('Your query is: ' + query)

    if(query == 'exit'):
        exit()

    variables = query.split(',')

    for i in range(len(variables)):
        variables[i] = variables[i].split('=')

    #Checks input for proper formatting
    acceptedAbbrevs = {"IW", "B", "R", "I", "SM", "G", "S", "M"}
    acceptedBools = {"true", "false"}

    for i in range(len(variables)):
        if(variables[i][0] not in acceptedAbbrevs):
            print("Please use the proper format, ABBREVIATION=bool")
            exit()
        if(variables[i][1] not in acceptedBools):
            print("Please use the proper format, ABBREVIATION=bool")
            exit()

    #print(joint_distribution(car_net))
    #print(set(all_rows(car_net)))

    #Convert the user's query into a boolean dictionary representing the known states
    listBools = {}

    for i in range(len(variables)):
        if(variables[i][0] == 'IW'):
            if(variables[i][1] == 'true'):
                listBools[IW] = T
            else:
                listBools[IW] = F
        elif(variables[i][0] == 'B'):
            if(variables[i][1] == 'true'):
                listBools[B] = T
            else:
                listBools[B] = F
        elif(variables[i][0] == 'SM'):
            if(variables[i][1] == 'true'):
                listBools[SM] = T
            else:
                listBools[SM] = F
        elif(variables[i][0] == 'R'):
            if(variables[i][1] == 'true'):
                listBools[R] = T
            else:
                listBools[R] = F
        elif(variables[i][0] == 'I'):
            if(variables[i][1] == 'true'):
                listBools[I] = T
            else:
                listBools[I] = F
        elif(variables[i][0] == 'G'):
            if(variables[i][1] == 'true'):
                listBools[G] = T
            else:
                listBools[G] = F
        elif(variables[i][0] == 'S'):
            if(variables[i][1] == 'true'):
                listBools[S] = T
            else:
                listBools[S] = F
        elif(variables[i][0] == 'M'):
            if(variables[i][1] == 'true'):
                listBools[M] = T
            else:
                listBools[M] = F

    #if('IW' not in listBools):
    #    listBools['IW'] = 'F'
    #if('B' not in listBools):
    #    listBools['B'] = 'F'
    #if('SM' not in listBools):
    #    listBools['SM'] = 'F'
    #if('R' not in listBools):
    #    listBools['R'] = 'F'
    #if('I' not in listBools):
    #    listBools['I'] = 'F'
    #if('G' not in listBools):
    #    listBools['G'] = 'F'
    #if('S' not in listBools):
    #    listBools['S'] = 'F'
    #if('M' not in listBools):
    #    listBools['M'] = 'F'

    #print(listBools)
    #print(enumeration_ask(IW, listBools, car_net))

    sumProbs = 0.0;

    for (row, p) in joint_distribution(car_net).items():
        if matches_evidence(row, listBools, car_net):
            sumProbs += p;

    print('The probability of your query is ' + str(sumProbs) + '\n')

#print(variables)
