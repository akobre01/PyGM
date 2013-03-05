from   Factor import Factor
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


