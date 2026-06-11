import torch
import numpy as np

class Softmax:
    '''
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))
        
        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:
        
        def __init__(self, weights_keys, weights_queries, weights_values):
            """
            Initialize instance variables. Refer to writeup for notation.
            input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

            Argument(s)
            -----------
            
            weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys
            weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries
            weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values
            """

            # Store the given weights as parameters of the class.
            self.W_k = weights_keys
            self.W_q = weights_queries
            self.W_v = weights_values

            # Use this object to perform softmax related operations.
            # It performs softmax over the last dimension which is what you'll need.
            self.softmax = Softmax()

        def forward(self, X):
            """
            Compute outputs of the self-attention layer.
            Stores keys, queries, values, raw and normalized attention weights.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            X (torch.tensor, dim = (B, T, D)): Input batch

            Return
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): Output batch
            """
            self.X = X
        
            # Compute the values of Key, Query and Value
            self.Q = X @ self.W_q
            self.K = X @ self.W_k
            self.V = X @ self.W_v

            # Calculate unormalized Attention Scores
            self.A_w = self.Q @ self.K.permute((0, 2, 1))

            # Create additive causal attention mask
            mask = torch.tril(torch.ones(self.A_w.shape)) == 0
            # apply mask
            self.A_w = torch.masked_fill(self.A_w, mask, -torch.inf)

            # Normalize Scaled Attention Scores
            self.D_k = self.Q.shape[2]
            self.A_sig = self.softmax.forward(self.A_w / np.sqrt(self.D_k))

            # Calculate Attention context
            X_new = self.A_sig @ self.V

            return X_new

        def backward(self, dLdXnew):
            """
            Backpropogate derivatives through the self-attention layer.
            Stores derivatives wrt keys, queries, values, and weight matrices.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

            Return
            ------
            dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs
            """
            # derivatives wrt attention weights (normalized and raw)
            dLdA_sig = dLdXnew @ self.V.permute((0, 2, 1))
            dLdA_w = self.softmax.backward(dLdA_sig) * 1 / np.sqrt(self.D_k)

            # derivatives wrt keys, queries, and value
            self.dLdK = dLdA_w.permute((0, 2, 1)) @ self.Q
            self.dLdQ = dLdA_w @ self.K
            self.dLdV = self.A_sig.permute((0, 2, 1)) @ dLdXnew

            # derivatives wrt weight matrices
            # sum derivatives along batch dimension.
            self.dLdWk = torch.einsum('bdt,btk->dk', self.X.permute((0, 2, 1)), self.dLdK)
            self.dLdWq = torch.einsum('bdt,btk->dk', self.X.permute((0, 2, 1)), self.dLdQ)
            self.dLdWv = torch.einsum('bdt,btv->dv', self.X.permute((0, 2, 1)), self.dLdV)

            # derivatives wrt input
            dLdX = self.dLdK @ self.W_k.T + self.dLdQ @ self.W_q.T + self.dLdV @ self.W_v.T

            return dLdX