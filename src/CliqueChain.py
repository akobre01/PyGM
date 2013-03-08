from   factor import Factor
import numpy  as     np

class CliqueChain:
    """
    Implements a clique tree that happens to be in a chain.  Right now,
    it requires the user to do a bunch of work now...hopefully it won't later
    """

    # pass in a list of the cliques IN ORDER (two cliques at adjacent indices
    # must have an edge between them
    def __init__(self, cliques):
        self._cliques = cliques

    # Pass in two factors that are going to pass messages
    # The first factor is the source of the message; the second is recipient
    def message(self, f1, f2):
        f1vars  = set(f1._vars)
        sepset  = set(f1._vars).intersection(set(f2._vars))
        message = f1.eliminate(f1vars - sepset)
        return message

    # returns a list of messages from C1 -> C2, C2 -> C3, etc.
    def forwardMessages(self):
        messages = [ self.message(self._cliques[0], self._cliques[1]) ]
        for i in range(1, len(self._cliques) - 1):
            updatedClique = self._cliques[i].add(messages[i-1])
            messages.append(self.message(updatedClique, self._cliques[i + 1]))
        return messages

    # returns a list of message from Cn -> Cn-1, Cn-1 -> Cn-2, etc...
    def backwardMessages(self):
        self._cliques.reverse()
        messages = self.forwardMessages()
        self._cliques.reverse()
        return messages

    # returns cluster beliefs
    def sumProduct(self):
        fwdMs = self.forwardMessages()
        bckMs = self.backwardMessages()
        bckMs.reverse()  # now: C2 -> C1, C3 -> C2, ...

        beliefs = [ self._cliques[0] ] + [ b.add(msg)
                                           for (b, msg)
                                           in zip(self._cliques[1:], fwdMs) ]

        beliefs = [ b.add(msg)
                    for (b, msg)
                    in zip(beliefs, bckMs) ] + [beliefs[-1] ]

        return beliefs

    # returns marginals over all variables in the model
    def allMarginals(self):
        beliefs = self.sumProduct()
        vars2beliefs = {}
        for b in beliefs:
            for v in b._vars:
                vars2beliefs[v] = b

        return [ b.marginal([var]).toProbs()
                 for (var,b) in vars2beliefs.items() ]

