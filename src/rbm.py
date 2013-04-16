from   factor import Factor
import numpy  as     np

class BinaryRBM:
    """
    Implements a binary RBM
    """

    def __init__(self, observed, nhidden, oparams = None, hparams = None,
                 pairParams = None):
        """
        pass in the number of hidden units; this method will initialize the
        (binary) hidden units randomly
         - oparams is a vector
         - hparams is a vector
         - pparams is a matrix

        I should do something smarter with the parameters
        """
        self._nhidden   = nhidden
        self._nobserved = np.size(observed)
        self._hidden    = np.random.randint(2, size=nhidden) # binary
        self._observed  = observed
        self._oparams   = oparams
        self._hparams   = hparams
        self._pparams   = pairParams

    def _obsGivenHidden(self, hidden = None):
        """
        Draw a sample from the probability of the observed variables given
        the hidden variables -- P(o | h).  This method assumes that the
        observations are binary and returns a binary vector
        """

        if hidden is None:
            hidden = self._hidden

        samples = np.random.rand(self._nobserved)
        z = np.exp(self._oparams + np.dot(self._pparams, hidden))

        probs = z / (1 + z)
        return ((probs - samples) >= 0).astype(float)

    def _hiddenGivenObs(self, observed = None):
        """
        Draw a sample from the probability of the hidden variables given the
        observed variables -- P(h | o).  This method assumes that the hidden
        variables are binary and returns a binary vector
        """

        if observed is None:
            observed = self._observed
        samples = np.random.rand(self._nhidden)
        z = np.exp(self._hparams + np.dot(self._pparams.T, observed))

        probs = z / (1 + z)
        return ((probs - samples) >= 0).astype(float)

    def blockGibbs(self, s):
        """
        pass in the number of iterations; thiscontinually draws
        samples of the hidden variables given the observed variables
        *AND THEN DRAWS NEW OBSERVED VARIABLES GIVEN A SAMPLE OF HIDDEN VARS*
        This algorithm uses block gibbs sampling
        """

        # Initialize
        oSamples = [ self._obsGivenHidden() ]
        hSamples = [ self._hiddenGivenObs(oSamples[-1]) ]

        for i in range(s-1):
            oSamples.append(self._obsGivenHidden(hSamples[-1]))
            hSamples.append(self._hiddenGivenObs(oSamples[-1]))

        return oSamples, hSamples
