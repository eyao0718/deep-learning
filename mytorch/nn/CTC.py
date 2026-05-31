import numpy as np

class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """
        self.blank = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        N = target.shape[0]
        extSymbols = np.full(2 * N + 1, self.blank, dtype=np.int64)
        extSymbols[1::2] = target
        skipConnect = np.zeros(2 * N + 1, dtype=np.int8)
        for i in range(len(skipConnect)):
            if i >= 2 and extSymbols[i] != extSymbols[i-2]:
                skipConnect[i] = 1
        return extSymbols, skipConnect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        input_len, _ = logits.shape
        ext_dim = extended_symbols.shape[0]
        alpha = np.zeros((input_len, ext_dim))
        alpha[0,0] = logits[0,extended_symbols[0]]
        alpha[0,1] = logits[0,extended_symbols[1]]
        for t in range(1, input_len):
            alpha[t,0] = alpha[t-1,0] * logits[t,extended_symbols[0]]
            for l in range(1, ext_dim):
                alpha[t,l] = alpha[t-1,l] + alpha[t-1,l-1]
                if skip_connect[l] == 1:
                    alpha[t,l] += alpha[t-1,l-2]
                alpha[t,l] *= logits[t,extended_symbols[l]]
        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
    
        """
        input_len, _ = logits.shape
        ext_dim = extended_symbols.shape[0]
        beta = np.zeros((input_len, ext_dim))
        beta[input_len-1,ext_dim-1] = logits[input_len-1,extended_symbols[ext_dim-1]]
        beta[input_len-1,ext_dim-2] = logits[input_len-1,extended_symbols[ext_dim-2]]
        for t in reversed(range(input_len-1)):
            beta[t,ext_dim-1] = beta[t+1,ext_dim-1] * logits[t,extended_symbols[ext_dim-1]]
            for l in reversed(range(ext_dim-1)):
                beta[t,l] = beta[t+1,l] + beta[t+1,l+1]
                if l+2 < ext_dim and skip_connect[l+2] == 1:
                    beta[t,l] += beta[t+1,l+2]
                beta[t,l] *= logits[t,extended_symbols[l]]
        ext_logits = np.zeros((input_len, ext_dim))
        for l in range(ext_dim):
            ext_logits[:,l] = logits[:,extended_symbols[l]]
        beta /= ext_logits
        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        gamma = alpha * beta
        gamma /= np.einsum('tl->t', gamma, optimize='optimal')[:, None]
        return gamma

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """
        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        """
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """
        CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """
        self.logits = logits
        self.t_logits = []
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []
        
        for b in range(B):
            t_target = target[b, :target_lengths[b]]
            t_logits = logits[:input_lengths[b], b, :]
            self.t_logits.append(t_logits)
            extSymbols, skipConnect = self.ctc.extend_target_with_blank(t_target)
            self.extended_symbols.append(extSymbols)
            alpha = self.ctc.get_forward_probs(t_logits, extSymbols, skipConnect)
            beta = self.ctc.get_backward_probs(t_logits, extSymbols, skipConnect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(gamma)
            div = 0
            for t in range(gamma.shape[0]):
                for l in range(gamma.shape[1]):
                    div += -1 * gamma[t, l] * np.log(t_logits[t, extSymbols[l]])
            total_loss[b] = div
        avgLoss = np.sum(total_loss) / B
        return avgLoss

    def backward(self):
        """

        CTC loss backward

        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """
        dY = np.zeros(self.logits.shape)
        B = self.logits.shape[1]
        for b in range(B):
            T, L = self.gammas[b].shape
            for t in range(T):
                for l in range(L):
                    dY[t, b, self.extended_symbols[b][l]] += -1 * self.gammas[b][t, l] / self.t_logits[b][t, self.extended_symbols[b][l]]
        return dY