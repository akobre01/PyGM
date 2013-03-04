import copy
import numpy as np

class Factor:
    """
    Represents a general factor in any type of factor graph
    """
    # make sure we have the ordering correct (this is CRUCIAL!)
    def _assertOrdering(self):
        for var,ind in self._vars2inds.iteritems():
            assert self._vars[ind] == var , "Incorrect internal ordering!"

    def _assertVarExists(self, v):
        assert self._vars2inds.has_key(v), "Factor doesn't touch " + str(v)

    def _assertTuple(self,t):
        assert type(t) == 'tuple', "must pass in a tuple"

    def _assertCorrectDimensions(self):
        for var, ind in self._vars2inds.iteritems():
            assert self._factor.shape[ind] == self._domains[var], (
                "Incorrect dimensionality")

    # return a copy of this factor
    def _copy(self):
        return Factor(copy.copy(self._vars),
                      self._domains.values(),
                      np.copy(self._factor))

    # swap dimensions SAFELY keeping all interal info correct (aka vars,
    # vars2inds, and factor itself
    # THIS NEEDS SOME ASSERTIONS
    def _safeDimSwap(self, frm, to):
        frmVar, toVar = self._vars[frm], self._vars[to]
        frmDom, toDom = self._domains[frmVar], self._domains[toVar]

        self._vars[frm], self._vars[to] = self._vars[to], self._vars[frm]
        self._vars2inds.update({ frmVar : to, toVar : frm})
        self._domains.update({ frmVar : toDom, toVar : frmDom })
        np.swapaxes(self._factor, frm, to)

    # safely add a dimension to the factor keeping track of all internal state
    # THIS NEEDS SOME ASSERTIONS
    def _safeDimAdd(self, var, ind):
        self._factor = np.expand_dims(self._factor, ind)
        self._vars.insert(ind, var)
        self._domains[var] = 1
        for v,i in self._vars2inds.iteritems():
            if i>= ind:
                self._vars2inds.update({v : i + 1})

    # After eliminating a var, reorganize indicies correctly by
    # subtracting 1 from indicies that are greater
    def _resetIndicies(self, index):
        for var, ind in self._vars2inds.iteritems():
            if ind > index:
                self._vars2inds.update({var : ind - 1})

    def __init__(self, variables, domains, factor):
        # make some assertions here
        self._factor    = factor
        self._vars      = variables
        self._domains   = {}
        self._vars2inds = {}
        for i,v in enumerate(variables):
            self._vars2inds[v] = i
            self._domains[v] = domains[i]

    # setting is a tuple; order matters!
    def set(self, setting, value):
        # make some assertions here
        self._assertTuple(setting)
        self._factor[setting] = value

    # setting is a tuple...perhaps think of converting to tuple if list
    def get(self, setting):
        self._assertTuple(setting)
        return self._factor[setting]

    # factor multiply (right now too complex)
    # strategy: 1) get the tensors to be the same dimensions using expand_dims
    #           2) make sure that the dimension ordering is the same for both
    #           3) broadcast the multiplication
    def multiply(self, newFactor):
        myFactorCopy  = self._copy()
        newFactorCopy = newFactor._copy()
        newVars = list(set(myFactorCopy._vars).union(set(newFactorCopy._vars)))

        for ind, var in enumerate(newVars):
            for factor in [myFactorCopy, newFactorCopy]:
                if factor._vars2inds.has_key(var):
                    factor._safeDimSwap(factor._vars2inds[var], ind)
                else:
                    factor._safeDimAdd(var, ind)

        _newFactor =  myFactorCopy._factor * newFactorCopy._factor
        return Factor(newVars, list(_newFactor.shape), _newFactor)

    # returns the order in which to specify factor settings
    def settingOrder(self):
        return self._vars

    # in place? Or should I return a new one?
    def observe(self, var, val):
        newFactor = self._factor.swapaxes(0,self._vars2inds[var])
        return newFactor

    # variable elimination
    def eliminate(self, var):
        if set([var]).issubset(set(self._vars)):
            varInd = self._vars2inds[var]
            self._factor = np.sum(self._factor, varInd)
            self._resetIndicies(varInd)
