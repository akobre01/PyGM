import numpy as np
from   CPT import CPT

class BayesNet:
    """Represents a Bayesian Network (or a joint distribution over CPTs)"""

    _vars2ind = {}
    _vars     = []
    _domains  = []
    _cpts     = []

    # takes list of variables and returns corresponding list of indicies
    def _indFromVar(self, variables):
        return [ self._vars2ind[v] for v in variables ]

    # takes list of variables and returns corresponding list of domains
    def _domainFromVar(self, variables):
        return [ self._domains[d] for d in self._indFromVar(variables) ]

    # variables is a list of all variables in a specific order (when passing
    #   in instances for learning, the values have to be in the same order)
    # domains is a list of those variables' domains IN THE SAME ORDER AS THE
    #   VARIBLES, and
    # conditions is a list of list of variables the first variable in each
    #   list contains the child, the others parents
    def __init__(self, variables, domains, conditionals):

        for d in domains:
            assert d > 0,       "All domains must be positive!"
            assert int(d) == d, "All domains must be integers!"

        assert len(set(variables)) == len(variables), "Error, duplicate vars!"
        assert set(variables) == set([x[0] for x in conditionals]), (
            "There must be 1 conditional for each variable!")
        assert len(variables) == len(domains), (
            "Number of variables and number of domain entries do not match!")

        self._vars2ind = dict([(e,i) for i,e in enumerate(variables)])
        self._vars     = variables
        self._domains  = domains
        self._cpts     = [ CPT(conditional, self._domainFromVar(conditional))
                           for conditional in conditionals ]

    # insts is a list of lists each of which contains a setting for each of
    # variables in this BayesNet
    def learn(self, insts):

        # quick function for getting relevant indices from instance
        def instSlice(indices, inst):
            return [ inst[i] for i in indices ]

        for inst in insts:
            assert len(inst) == len(self._vars), (
                "Instance " + str(inst) + " should be of length " +
                str(len(self.vars)))

            for i,v in enumerate(inst):
                assert v >= 0 and v < self._domains[i], (
                    "Value " + str(v) + " is not within correct domain!")

        # Get relevant indicies for each CPT
        cptSlicers = [ self._indFromVar(cpt.getVars()) for cpt in self._cpts ]

        # Build a list of instances for each CPT to learn by slicing out
        # the correct incides from each training instance
        learningSets = [[ instSlice(cslice, inst) for inst in insts ]
                        for cslice in cptSlicers ]

        [ cpt.learn(x) for (cpt, x) in zip(self._cpts, learningSets) ]

    def showJoint(self):
        lval = "P(" + ",".join(self._vars) + ")"
        rval = "".join([cpt.getConditional() for cpt in self._cpts])
        print(lval + " = " +rval)

    def showAllCPTs(self):
        for cpt in self._cpts:
            cpt.showProbTable()
