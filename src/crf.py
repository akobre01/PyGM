from   scipy       import optimize as opt
from   factor      import Factor
from   cliqueChain import CliqueChain
import numpy  as     np

from config import CHARS, CHARS_DICT

class CRF:
    """
    Implements a CRF -- more later
    """

    # This is a discrete state CRF...fill this out
    # Weights is an NxM numpy array:
    #   N - number of different possible labels
    #   M - the number of features (each feature should be binary)
    # transProbs is an NxN numpy array:
    #   N - as above, N is the number of different possible labels
    def __init__(self, weights, transProbs, trainInstances, trainLabels):
        assert weights.shape[0] == transProbs.shape[0], (
            "weights.shape[0] MUST equal transProbs.shape[0]")
        assert transProbs.shape[0] == transProbs.shape[1], (
            "transProbs.shape[0] MUST equal transProbs.shape[1]")

        self._instances = trainInstances
        self._labels    = trainLabels
        self._numLabels = weights.shape[0]
        self._numFeats  = weights.shape[1]
        self._ws        = weights
        self._tps       = transProbs
        self._VAR_PREF  = 'label'

    def _label(self, i):
        assert type(i) is int
        return self._VAR_PREF + str(i)

    # turn an instance into a cliqueChain
    def _instance2Chain(self, instance, weights, transProbs):
        assert instance.shape[1] == self._numFeats, (
            "Instance passed in must have " + self._numFeats + " features!")

        potentials = np.dot(weights, np.transpose(instance))
        labelFctrs = [ Factor([self._label(i)],
                              [ self._numLabels ],
                              potentials[:,i])
                       for i in range(instance.shape[0])]
        pairwiseFctrs = [ Factor([self._label(i), self._label(i+1)],
                                 [self._numLabels, self._numLabels],
                                 transProbs)
                          for i in range(instance.shape[0] - 1)]

        # Build Clique Chain
        cliques = [ labelFctr.add(pairwiseFctr)
                    for (labelFctr, pairwiseFctr)
                    in zip(labelFctrs, pairwiseFctrs) ]
        cliques[-1] = cliques[-1].add(labelFctrs[-1])
        return CliqueChain(cliques)

    # logLikelihood of an instance given the current setting of the weights
    def instanceLogLikelihood(self, instance, assignment, weights, transProbs):
        chain = self._instance2Chain(instance, weights, transProbs)
        return chain.logLikelihood(assignment)

    # get the average log likelihood for a group of instances and assignments
    def avgLogLikelihood(self, instances, labels, weights, transProbs):
        return sum([ self.instanceLogLikelihood(i,a,weights,transProbs)
                     for i,a in zip(instances, labels)
                     ]) / float(len(instances))

    # NEEDS BETTER DOCUMENTATION!!!!
    # derivative of the object function (log likelihood) with respect to the
    # feature weights (produces a NxM vector)
    def _labelWeightsDerivHelper(self, inst, marginals, varNames, labels):
        actualMinusExpectedCounts = np.zeros((self._numLabels, self._numFeats))

        for i,var in enumerate(varNames):
            counts = np.zeros(self._numLabels)
            counts[labels[var]] += 1.0
            counts -= marginals[var]._factor

            actualMinusExpectedCounts += (counts[:,np.newaxis] * inst[i])

#            actualMinusExpectedCounts[labels[var]] += (
#                ( np.ones(self._numFeats) -
#                  marginals[var]._factor[labels[var]] ) *
#                inst[i] )

#        print(actualMinusExpectedCounts.shape)
#        print(np.sum(actualMinusExpectedCounts))
        return actualMinusExpectedCounts

    # MUST DOCUMENT BETTER!
    # derivative of the object function (log likelihood) with respect to the
    # transition probabilities (produces an MxM vector)
    def _transProbsDerivHelper(self, pairMarg, varNames, labels):
        actualMinusExpectedCounts = np.zeros((self._numLabels,self._numLabels))

        # commence numpy voodoo
        transitions = [ (labels[v1], labels[v2])
                        for (v1,v2) in zip(varNames, varNames[1:]) ]

        relevantMargs = np.array([ pairMarg[v]._factor[t1][t2]
                                   for (v, (t1,t2))
                                   in zip(varNames, transitions) ])

#        updates = np.ones(len(relevantMargs)) - np.array(relevantMargs)
#        for i,update in enumerate(updates):
#            (v1,v2) = transitions[i]
#            actualMinusExpectedCounts[v1][v2] += 1.0

        for i,v in enumerate(varNames[:-1]):
            (v1,v2) = transitions[i]
            actualMinusExpectedCounts[v1][v2] += 1.0
            actualMinusExpectedCounts -= pairMarg[v]._factor

#        print(actualMinusExpectedCounts)
#        print(np.sum(actualMinusExpectedCounts))
        return actualMinusExpectedCounts

    # jacobian of the objective; this returns a long vector of all params
    def _jacobian(self, weightsAndTransProbs):
        nL, nF = self._numLabels, self._numFeats
        weights    = np.reshape(weightsAndTransProbs[:(nL * nF)], (nL, nF))
        transProbs = np.reshape(weightsAndTransProbs[(nL * nF):], (nL, nL))

        updatedWeights = np.zeros((self._numLabels, self._numFeats))
        updatedTransP  = np.zeros((self._numLabels, self._numLabels))

        for inst, labels in zip(self._instances, self._labels):
            varNames = [ self._label(i) for i in range(len(inst)) ]
            chain    = self._instance2Chain(inst, weights, transProbs)

            # Pairwise marginals - the min trick picks label0 over label1, etc.
            beliefs  = chain.sumProduct()
            pairMarg = dict(map(lambda x: (min(x._vars), x.toProbs()),
                                beliefs))


            # Singleton marginals - find a way to simplify
            vars2beliefs = {}
            for b in beliefs:
                for v in b._vars:
                    vars2beliefs[v] = b
            marginals = dict([ (var, b.marginal([var]).toProbs())
                               for (var,b) in vars2beliefs.items() ])

            updatedWeights += self._labelWeightsDerivHelper(inst,
                                                            marginals,
                                                            varNames,
                                                            labels)
            updatedTransP  += self._transProbsDerivHelper(pairMarg,
                                                          varNames,
                                                          labels)

        return -np.concatenate((np.reshape(updatedWeights, nL * nF),
                                np.reshape(updatedTransP, nL * nL)),
                               axis=0) / float(len(self._instances))

    # combines arguments of objective function for easy optimizing with scipy
    def _objective(self, weightsAndTransProbs):
        J = weightsAndTransProbs.copy()
        nL, nF     = self._numLabels, self._numFeats
        weights    = np.reshape(J[:(nL * nF)], (nL, nF))
        transProbs = np.reshape(J[(nL * nF):], (nL, nL))

        # want to maximize the log likelihood/minimize negative log likelihood
        toReturn = -self.avgLogLikelihood(self._instances,
                                          self._labels,
                                          weights,
                                          transProbs)
        print(toReturn)
        return toReturn

    # train this CRF on the instances passed in
    def train(self, instances, labels):
        self._instances = instances
        self._labels    = labels
        nL, nF = self._numLabels, self._numFeats

        objective = lambda x: self._objective(x)
        jacobian  = lambda x: self._jacobian(x)
        result    = opt.minimize(objective,
                                 np.random.normal(0,1,nL * nF + nL * nL),
                                 jac=jacobian,
                                 method="BFGS")

        learnedLabelWeights = np.reshape(result.x[:(nL * nF)], (nL, nF))
        learnedTransProbs   = np.reshape(result.x[(nL * nF):], (nL, nL))

        self._ws  = learnedLabelWeights
        self._tps = learnedTransProbs
        return (learnedLabelWeights, learnedTransProbs)

