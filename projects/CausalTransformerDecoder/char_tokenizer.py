class CharTokenizer():

    '''
    A wrapper around character tokenization to have
    a consistent interface with other tokenization strategies
    '''

    def __init__(self):
        # Same as EOS_TOKEN
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Same ID as EOS_TOKEN
        self.bos_token_id = self.vocab[self.eos_token]
        
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)
    
    # convert string to elements in the vocab
    def tokenize(self, data):
        return [char for char in data]
    
    # convert string to list of id
    def encode(self, data, return_tensors=None):
        e = [self.vocab.get(key=char.upper(), default=self.unk_token_id) for char in data]
        if return_tensors == 'pt':
            return torch.tensor(e).unsqueeze(0)
        return e
    
    # convert ids to list of char
    def decode(self, data):
        try:
            return ''.join([self.inv_vocab.get(j) for j in data])
        except:
            data = data.cpu().tolist()
            return ''.join([self.inv_vocab.get(j) for j in data])

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]