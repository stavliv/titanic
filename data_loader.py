from numpy import iterable


class DataLoader():
    '''
    a DataLoader

    Parameters
    ----------
    dataset : list (iterbale)
        the dataset in the form [inputs, labels],
        inputs torch.Tensor(shape=(#training examples, #input features)),
        labels torch.Tensor(shape=(#training examples, #output features))
    batch_size : int
        batch size

    Attributes
    ----------
    dataset : list (iterbale)
        the dataset in the form [inputs, labels],
        inputs torch.Tensor(shape=(#training examples, #input features)),
        labels torch.Tensor(shape=(#training examples, #output features))
    batch_size : int
        batch size
    '''
    def __init__(self, dataset: list, batch_size: int):
        self.dataset, self.batch_size = dataset, batch_size
    def __iter__(self):
        '''

        Yields
        ----------
        list
            the next batch of the dataset in the form [inputs, labels]
        '''
        for i in range(0, len(self.dataset[0]), self.batch_size): yield (self.dataset[0][i:i+self.batch_size], self.dataset[1][i:i+self.batch_size])