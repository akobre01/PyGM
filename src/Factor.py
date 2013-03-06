import copy
import numpy as np
from scipy.misc import logsumexp

class Factor:
    """
    Represents a general log factor in any type of factor graph
    """
    # make sure we have the ordering correct (this is CRUCIAL!)
    def _assertOrdering(self):
        for var,ind in self._vars2inds.iteritems():
            assert self._vars[ind] == var , "Incorrect internal ordering!"

    def _assertVarExists(self, v):
        assert self._vars2inds.has_key(v), "Factor doesn't touch " + str(v)

    def _assertTuple(self,t):
        assert type(t) == 'tuple', "must pass in a tuple"

    def _assertValWithinDomain(self, var, val):
        self._assertVarExists(var)
        assert self._domains[var] > val, (
            str(val) + " is outside the valid domain of var: " + str(var))

    def _assertCorrectDimensionality(self):
        for var, ind in self._vars2inds.iteritems():
            assert self._factor.shape[ind] == self._domains[var], (
                "Incorrect dimensionality")

    # return a copy of this factor
    def _copy(self):
        return Factor(copy.copy(self._vars),
                      self._domains.values(),
                      np.copy(self._factor))

    # swap dimensions SAFELY keeping all internal state correct (aka vars,
    # vars2inds, and factor itself
    def _safeDimSwap(self, frm, to):
        frmVar, toVar = self._vars[frm], self._vars[to]
        frmDom, toDom = self._domains[frmVar], self._domains[toVar]

        self._vars[frm], self._vars[to] = self._vars[to], self._vars[frm]
        self._vars2inds.update({ frmVar : to, toVar : frm})
        self._domains.update({ frmVar : toDom, toVar : frmDom })
        np.swapaxes(self._factor, frm, to)
        self._assertCorrectDimensionality()
        self._assertOrdering()

    # safely add a dimension to the factor keeping track of all internal state
    def _safeDimAdd(self, var, ind):
        self._factor = np.expand_dims(self._factor, ind)
        self._vars.insert(ind, var)
        self._domains[var] = 1
        for v,i in self._vars2inds.iteritems():
            if i>= ind:
                self._vars2inds.update({v : i + 1})
        self._assertCorrectDimensionality()
        self._assertOrdering()

    # After eliminating a var, reorganize indicies correctly by
    # subtracting 1 from indicies that are greater
    def _resetIndicies(self, index):
        for var, ind in self._vars2inds.iteritems():
            if ind > index:
                self._vars2inds.update({var : ind - 1})

    # It's a bit weird here that you pass in a list for domains and it gets
    # converted to a dictionary...especially because variables is a dict
    # ALSO MAKE SURE THAT factor THAT IS PASSED IN IS IN LOG-SPACE!
    def __init__(self, variables, domains, factor):
        # make some assertions here
        self._factor    = factor
        self._vars      = variables
        self._domains   = {}
        self._vars2inds = {}
        self._assigned  = {}
        for i,v in enumerate(variables):
            self._vars2inds[v] = i
            self._domains[v] = domains[i]
        self._assertOrdering()
        self._assertCorrectDimensionality()

    # get current ordering of the variables
    def varOrdering(self):
        return self._vars

    # setting is a tuple; order matters!
    def set(self, setting, value):
        # make some assertions here
        self._assertTuple(setting)
        self._factor[setting] = value

    # setting is a tuple...perhaps think of converting to tuple if list
    def get(self, setting):
        self._assertTuple(setting)
        return self._factor[setting]

    # visual representation
    def show(self):
        assigned = []
        for var, assignment in self._assigned.iteritems():
            assigned.append(str(var) + " = " + str(assignment))
        return "F(" + ",".join(self._vars + assigned) + ")"

    # log-factor add (factor multiply)
    # strategy: 1) get the tensors to be the same dimensions using expand_dims
    #           2) make sure that the dimension ordering is the same for both
    #           3) broadcast the multiplication
    # (right now too complex)
    def add(self, newFactor):
        myFactorCopy  = self._copy()
        newFactorCopy = newFactor._copy()
        newVars = list(set(myFactorCopy._vars).union(set(newFactorCopy._vars)))

        for ind, var in enumerate(newVars):
            for factor in [myFactorCopy, newFactorCopy]:
                if factor._vars2inds.has_key(var):
                    factor._safeDimSwap(factor._vars2inds[var], ind)
                else:
                    factor._safeDimAdd(var, ind)

        _newFactor =  myFactorCopy._factor + newFactorCopy._factor
        return Factor(newVars, list(_newFactor.shape), _newFactor)

    # observe a single variable; updates vars, vars2inds, domains and
    # adds an entry to assignment
    # NEEDS ASSERTIONS AND MORE CHECKING
    def observe(self, var, val):
        newFactor = self._copy()
        newFactor._assertVarExists(var)
        newFactor._assertValWithinDomain(var,val)
        dim = newFactor._vars2inds.pop(var)
        newFactor._domains.pop(var)
        newFactor._vars.pop(dim)
        newFactor._assigned[var] = val
        newFactor._safeDimSwap(dim, 0)
        newFactor._factor = newFactor._factor[val,:]
        return newFactor

    # variable elimination; eliminates a single variable and does all
    # necessary state checking;
    # change this so that it works IN PLACE (impure) so it doesn't blow stack
    def _eliminate(self, var):
        newFactor = self._copy()
        if set([var]).issubset(set(newFactor._vars)):
            varInd = newFactor._vars2inds.pop(var)
            newFactor._domains.pop(var)
            newFactor._vars.pop(varInd)
            newFactor._factor = logsumexp(newFactor._factor, varInd)
            newFactor._resetIndicies(varInd)
            newFactor._assertCorrectDimensionality()
            newFactor._assertOrdering()
        return newFactor

    # pass in a list of vars to be elimated and call eliminate on each in turn
    def eliminate(self, vs):
        if len(vs) == 0:
            return self
        else:
            newFactor = self._eliminate(vs.pop())
            return newFactor.eliminate(vs)

    # return a marginalized log factor (NOT PROBABILITY) over specific vars
    # this is similar to eliminate but takes the variable you want to keep
    # instead of those you want to eliminate
    def marginal(self, vs):
        newFactor = self._copy()
        for v in vs:
            newFactor._assertVarExists(v)
        allVars = set(newFactor._vars)
        toEliminate = allVars - set(vs)
        return newFactor.eliminate(toEliminate)

    # exponentiate and then normalize this factor to return probabilities
    def toProbs(self):
        newFactor = self._copy()
        newFactor._factor = np.exp(newFactor._factor - newFactor.logZ())
        return newFactor # NO LONGER IN LOG SPACE!

    # calcualte the log of the partition function
    def logZ(self):
        toReturn = self._factor
        for dim in range(self._factor.ndim):
            toReturn = logsumexp(toReturn, 0)
        return toReturn
