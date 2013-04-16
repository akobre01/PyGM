from   factor import Factor
import numpy  as     np

class BinaryRBM:
    """
    Implements a binary RBM
    """

    def __init__(self, nobserved, nhidden, oparams = None, hparams = None,
                 pairParams = None):
        """
        pass in the number of hidden units; this method will initialize the
        (binary) hidden units randomly
         - oparams is a vector
         - hparams is a vector
         - pparams is a matrix

        I should do something smarter with the parameters
        """

        self._nobserved = nobserved
        self._nhidden   = nhidden
        self._oparams   = oparams
        self._hparams   = hparams
        self._pparams   = pairParams

    def _pObsGivenHidden(self, hidden):
        """
        Return the probability of the observed variables each taking
        the value 1 given the hidden variables
        """

        z = np.exp(self._oparams + np.dot(self._pparams, hidden.T).T)
        return z / (1 + z)

    def _pHiddenGivenObs(self, observed):
        """
        Return the probability of the hidden variables each taking
        the value 1 given the observed variables
        """

        z = np.exp(self._hparams + np.dot(self._pparams.T, observed.T).T)
        return z / (1 + z)

    def _obsGivenHidden(self, hidden):
        """
        Draw a sample from the probability of the observed variables given
        the hidden variables -- P(o=1| h).  This method assumes that the
        observations are binary and returns a binary vector
        """

        probs   = self._pObsGivenHidden(hidden)
        samples = np.random.rand(np.size(probs,0), np.size(probs,1))
        return ((probs - samples) >= 0).astype(int)

    def _hiddenGivenObs(self, observed):
        """
        Draw a sample from the probability of the hidden variables given the
        observed variables -- P(h=1| o).  This method assumes that the hidden
        variables are binary and returns a binary vector
        """

        probs   = self._pHiddenGivenObs(observed)
        samples = np.random.rand(np.size(probs,0), np.size(probs,1))
        return ((probs - samples) >= 0).astype(int)

    def blockGibbs(self, s):
        """
        pass in the number of iterations; thiscontinually draws
        samples of the hidden variables given the observed variables
        *AND THEN DRAWS NEW OBSERVED VARIABLES GIVEN A SAMPLE OF HIDDEN VARS*
        This algorithm uses block gibbs sampling
        """

        # Initialize
        oSamples = [ self._obsGivenHidden(
                np.random.randint(2, size=self._nhidden)) ]
        hSamples = [ self._hiddenGivenObs(oSamples[-1]) ]

        for i in range(s-1):
            oSamples.append(self._obsGivenHidden(hSamples[-1]))
            hSamples.append(self._hiddenGivenObs(oSamples[-1]))

        return oSamples, hSamples

    def learn(self, train, T, B, C, alpha, reg):
        """
        Trains the weights of a binary rbm; learning done using batch
        stochastic gradient ascent
        - train is the training set
        - T, number of learning iterations
        - C, number of chains to run
        - B, number of batches
        - alpha, learning rate
        - reg, regularization
        """

        ntrain      = np.size(train, 0)
        batchSize   = np.floor(ntrain / B).astype(int)
        batchStarts = np.append(np.arange(0, ntrain, batchSize), ntrain)

        hidSamps = np.random.randint(2, size=(C, self._nhidden))
        obsSamps = np.zeros((C, self._nobserved))

        # initiliaze params
        self._hparams = np.random.normal(0, 0.1**2, self._nhidden)
        self._oparams = np.random.normal(0, 0.1**2, self._nobserved)
        self._pparams = np.random.normal(0, 0.1**2, (self._nobserved,
                                                     self._nhidden))
        for t in range(T):
            for b in range(np.size(batchStarts) - 1):
                print("Iteration: " + str(t) + " Batch number: " + str(b))
                currBatch = train[batchStarts[b] : batchStarts[b+1], :]

                # Calculate the positive gradient pieces
                p_gOparams = np.sum(currBatch, 0)
                p_gHparams = np.zeros(self._nhidden)
                p_gPparams = np.zeros((self._nobserved, self._nhidden))

                probs      = self._pHiddenGivenObs(currBatch)
                p_gHparams = np.sum(probs, 0)
                p_gPparams = p_gPparams + np.dot(currBatch.T, probs)

                # for i in range(np.size(currBatch,0)):
                #     probs      = self._pHiddenGivenObs(currBatch[i])
                #     p_gHparams = p_gHparams + probs
                #     p_gPparams = p_gPparams + np.outer(currBatch[i], probs)

                # Calculate the negative gradient pieces
                n_gOparams = np.zeros(self._nobserved)
                n_gHparams = np.zeros(self._nhidden)
                n_gPparams = np.zeros((self._nobserved, self._nhidden))

                obsSamps = self._obsGivenHidden(hidSamps)
                hidSamps = self._hiddenGivenObs(obsSamps)

                n_gOparams = np.sum(obsSamps, 0)
                probs      = self._pHiddenGivenObs(obsSamps)
                n_gHparams = np.sum(probs, 0)
                n_gPparams = np.dot(obsSamps.T, probs)

                # for i in range(C):
                #     obsSamps[i] = self._obsGivenHidden(hidSamps[i])
                #     hidSamps[i] = self._hiddenGivenObs(obsSamps[i])

                #     n_gOparams = n_gOparams + obsSamps[i]
                #     probs      = self._pHiddenGivenObs(obsSamps[i])
                #     n_gHparams = n_gHparams + probs
                #     p_gPparams = p_gPparams + np.outer(obsSamps[i], probs)

                # Take Gradient Steps
                self._oparams = (self._oparams +
                                 alpha * (p_gOparams / np.size(currBatch, 0)
                                          - n_gOparams / C
                                          - reg * self._oparams))

                self._hparams = (self._hparams +
                                 alpha * (p_gHparams / np.size(currBatch, 0)
                                          - n_gHparams / C
                                          - reg * self._hparams))

                self._pparams = (self._pparams +
                                 alpha * (p_gPparams / np.size(currBatch, 0)
                                          - n_gPparams / C
                                          - reg * self._pparams))

        return obsSamps, hidSamps


