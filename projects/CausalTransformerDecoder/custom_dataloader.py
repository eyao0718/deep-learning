# Inherit from torch.utils.data.DataLoader
class DataLoaderForLanguageModeling(torch.utils.data.DataLoader):
    """
    Data loader logic
    """

    def __init__(self, dataset, batch_size, sequence_length=3, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sequence_length = sequence_length
        self.concatenated_dataset = None
        self.num_batches = 0

    def __len__(self):
        """
        Concatenate the dataset and then batch parts of it according to the sequence length.
        If you are using variable sequence_length, the length might not be fixed.

        Returns:
            num_batches: total number of batches
        """
        self.concatenated_dataset = np.concatenate(self.dataset, axis=0)
        dropped_length = self.concatenated_dataset.shape[0] // self.sequence_length * self.sequence_length
        num_seq = dropped_length // self.sequence_length
        self.num_batches = num_seq // self.batch_size
        if self.drop_last == False and num_seq % self.batch_size != 0:
            self.num_batches += 1
        return self.num_batches

    def __iter__(self):
        """
        Returns:
            input_batch, target_batch: torch.tensor(self.batch_size, self.sequence_length)
        """

        # shuffle data if shuffle=True
        if self.shuffle:
            instance = np.random.default_rng()
            instance.shuffle(self.dataset, axis=0)

        # concatenate newly shuffled dataset
        self.concatenated_dataset = np.concatenate(self.dataset, axis=0)

        # drop extra words of sequences
        dropped_length = self.concatenated_dataset.shape[0] // self.sequence_length * self.sequence_length
        self.concatenated_dataset = self.concatenated_dataset[:dropped_length]

        # create inputs and shifted targets through window slicing
        inputs, targets = [], []
        for i in range(0, dropped_length - self.sequence_length, self.sequence_length):
            input_seq = self.concatenated_dataset[i:i+self.sequence_length]
            inputs.append(input_seq)
            target_seq = self.concatenated_dataset[i+1:i+1+self.sequence_length]
            targets.append(target_seq)

        # array of arrays of sequence ids
        inputs = np.array(inputs)
        targets = np.array(targets)
        # pad for incomplete last batch
        if self.drop_last == False and len(inputs) % self.batch_size != 0:
            padding_size = self.batch_size - (len(inputs) % self.batch_size)
            pad = np.full((padding_size, self.sequence_length), PAD_TOKEN)
            inputs = np.concatenate((inputs, pad), axis=0)
            targets = np.concatenate((targets, pad), axis=0)
        # drop incomplete last batch
        else:
            new_length = len(inputs) // self.batch_size * self.batch_size
            inputs = inputs[:new_length]
            targets = targets[:new_length]

        # reshape for batch processing
        self.num_batches = len(inputs) // self.batch_size
        inputs = np.reshape(inputs, (self.num_batches, self.batch_size, self.sequence_length))
        targets = np.reshape(targets, (self.num_batches, self.batch_size, self.sequence_length))
        
        # yield each batch
        for b in range(self.num_batches):
            input_batch = inputs[b]
            target_batch = targets[b]
            yield torch.tensor(input_batch, dtype=torch.long), torch.tensor(target_batch, dtype=torch.long)