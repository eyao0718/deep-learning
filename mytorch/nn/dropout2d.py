import numpy as np


class Dropout2d(object):

    def __init__(self, p=0.5):
       
       self.p = p
       
    def __call__(self, *args, **kwargs):
       
       return self.forward(*args, **kwargs)
    
    def forward(self, x, eval=False):
        """
        Arguments:
        x (np.array): (batch_size, in_channel, input_width, input_height)
        eval (boolean): whether the model is in evaluation mode
        Return:
        np.array of same shape as input x
        """
        batch_size, in_channel, _, _ = x.shape

        if eval:
            return x
        
        else:

            self.mask = np.random.binomial(n=1, p=1-self.p, size=(batch_size,in_channel))
            self.mask = self.mask[:,:,None,None]

            self.alpha = 1 / (1 - self.p)

            return self.alpha * self.mask * x
        
    def backward(self, delta):
        """
        Arguments:
        delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
        np.array of same shape as input delta
        """
        
        return self.alpha * self.mask * delta