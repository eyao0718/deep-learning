import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        Initialize instance variables

        Argument(s)
        -----------
        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)
        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        Perform greedy search decoding

        Input
        -----
        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]

        Returns
        -------
        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path
        """

        L, T, B = y_probs.shape
        decoded_path = []
        blank = 0
        path_prob = np.ones(B)

        for b in range(B):

            subpath = []

            for t in range(T):
                path_prob[b] *= np.max(y_probs[:,t,b])
                idx = np.argmax(y_probs[:,t,b])
                if idx == blank:
                    subpath.append('-')
                else:
                    subpath.append(self.symbol_set[idx-1])

            comp_subpath = []

            if subpath[0] != '-':
                comp_subpath.append(subpath[0])
            for c in range(1, len(subpath)):
                if subpath[c] != '-' and subpath[c] != subpath[c-1]:
                    comp_subpath.append(subpath[c])

            decoded_path.append(comp_subpath)

        decoded_path = ["".join(decoded_path[i]) for i in range(len(decoded_path))]

        return decoded_path[0], path_prob[0]

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables

        Argument(s)
        -----------
        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion
        """
        self.symbol_set = ['-'] + symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        Perform beam search decoding

        Input
        -----
        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]

        Returns
        -------
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores
        """
        B = y_probs.shape[2]
        forward_path = []
        MergedPathScores = []
        for b in range(B):
            forward_path.append("-")
            MergedPathScores.append({})
            _, T = y_probs[:, :, b].shape
            BestPaths = {}
            BestPaths["-"] = 1.0
            for t in range(T):
                prob_t = y_probs[:, t, b]
                TempBestPaths = {}
                for path, score in BestPaths.items():
                    for l, pr in enumerate(prob_t):
                        new_path = path
                        extChar = self.symbol_set[l]
                        if path[-1] == "-" or path[-1] == extChar:
                            new_path = new_path[:-1] + extChar
                        else:
                            new_path = new_path + extChar
                        if new_path in TempBestPaths:
                            TempBestPaths[new_path] += score * pr
                        else:
                            TempBestPaths[new_path] = score * pr
                # Super Important!
                # Don't prune TempBestPaths at last time step cuz
                # you don't plan to expand tree anyways afterwards
                if t != T-1 and len(TempBestPaths) > self.beam_width:
                    BestPaths = dict(sorted(TempBestPaths.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])
                else:
                    BestPaths = TempBestPaths
            BestScore = 0.0
            for path, score in BestPaths.items():
                if path[-1] == "-":
                    path = path[:-1]
                if path in MergedPathScores[b]:
                    MergedPathScores[b][path] += score
                else:
                    MergedPathScores[b][path] = score
                if MergedPathScores[b][path] > BestScore:
                    forward_path[b] = path
                    BestScore = MergedPathScores[b][path]
        return forward_path[0], MergedPathScores[0]