import numpy as np
from   numpy import cumprod
from   numpy import array

class CPT:
    """Represents a conditional probability table over a set of discrete
    random variables.  There might seem like there is a lot of magic going
    on here. It's beause of the representation and calculating indices of
    things"""

    _vars2inds   = {}  # maps variables to indicies; index 0 is the child
    _vars        = []  # we must maintain the invariant that:
    _domains     = []  #    dom(vars[i]) == domains[i] and that
    _probs       = []  # probabilities of various variable configurations

    # THINK ABOUT THIS CONSTRUCTOR
    # pass in 2 dictionaries; each one maps variable names to number of
    # values that variable can take on.  The first dictionary corresponds
    # to the child node that will be conditional on its parents


    # pass in 2 lists: 1) of variables included in this CPT and
    #                  2) the domains of those variables IN ORDER
    # by convention, the first variable is the child, the rest are parents
    def __init__(self, variables, domains):
        assert len(set(variables)) == len(variables), ("Duplicate variables " +
                                                       "submitted!")
        assert len(variables) == len(domains), ("Number of variables and " +
                                                "number of domain entries " +
                                                "do not match!")
        for d in domains:
            assert d > 0, "Domains must all be greater than 0!"
            assert int(d) == d, ("Domains must all be integers!")

        self._domains = domains
        self._probs   = np.zeros(np.product(self._domains))
        self._vars    = variables
        self._vars2ind = dict([ (pair[1], pair[0])
                                for pair in enumerate(variables) ])

    # make sure that a particular setting of the variables checks out
    def _assertSetting(self, setting, domain):
        assert len(setting) == len(domain)
        for i,e in enumerate(setting):
            assert e >= 0,        "All values must be >= 0!"
            assert e < domain[i], "All values must be < domain!"
            assert int(e) == e,   "Settings must all be integers!"

    # setting is a list of settings. ORDER MATTERS! The ordering must be
    # the same ordering as the ordering of both _domains and _vars
    def _setting2Index(self, setting):
        offsetsArray = np.append(1, cumprod(self._domains))
        offsetsArray = offsetsArray.take(range(0,len(self._domains)))
        return np.dot(array(setting),offsetsArray)

    # return the probability of a particular setting
    def probs(self, setting):
        return self.probs[self._setting2Index(setting)]

    # examples is a list of settings of the parameter (each setting is itself
    # a list.  Fill in the probability table in the internal CPT
    def learn(self, examples):
        for ex in examples:
            self._assertSetting(ex, self._domains)

        self._probs = np.zeros(np.product(self._domains))
        for ex in examples:
            self._probs[self._setting2Index(ex)] += 1

        self._normalizeCPT()

    def _normalizeArray(self, a):
        return a / float(sum(a))

    # normalize this CPT: for every configuartion of the parents, the
    # sum of the probabilities of changing the child setting is 1
    def _normalizeCPT(self):
        splitSize = np.product(self._domains) / self._domains[0]
        splits    = np.split(self._probs, splitSize)
        # splits now contains arrays of size _domains[0] that we'll normalize

        normalized  = array([ self._normalizeArray(a) for a in splits ])
        self._probs = normalized.flatten()

    # create list of settings of variables with the following domains
    # this is a GROSS function
    def _enumerateSettings(self, domains):
        numRows   = np.product(domains)
        numRepeat = 1
        numTile   = numRows
        rows = []
        for d in domains:
            numTile /= d
            row = np.tile(array(range(d)).repeat(numRepeat), numTile)
            numRepeat *= d
            rows.append(row)

        return array(rows).transpose()

    def getVars(self):
        return self._vars

    # Return a visual representation of the conditional disribution
    # represented by this CPT
    def getConditional(self):
        parents = ""
        if len(self._vars) > 1:
            parents = "|" + ",".join([str(x) for x in self._vars[1:]])
        return "P(" + self._vars[0] + parents + ")"

    def showConditional(self):
        print(self.getConditional())

    def showProbTable(self):
        settings = self._enumerateSettings(self._domains)
        print(" ".join([str(x) for x in self._vars]) +
              " " + self.getConditional())

        for i,e in enumerate(settings):
            setting = [int(x) for x in e.tolist()]
            setting.append(self._probs[i])
            print("|".join([str(x) for x in setting]))
