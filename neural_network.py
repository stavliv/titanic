import torch
from plot import plot_learning_curve

class NeuralNetwork():
    '''
    a neural network

    Parameters
    ----------
    model : Any
        the model to train,
    optimizer : Any
        the optimizer used in training
    criterion : Any
        the loss function used in training
    metric_funcs : dict
        the metrics we want to compute, each key is a string with the metrics' name and value
        the function to compute it. 
        For every function it must be: function(predictions, labels) -> metric
    device : torch.device
        the device used for the training
    tag : str
        the nametag of the neural network


    Attributes
    ----------
    model : Any
        the model to train,
    optimizer : Any
        the optimizer used in training
    criterion : Any
        the loss function used in training
    metric_funcs : dict
        the metrics we want to compute, each key is a string with the metrics' name and value
        the function to compute it. 
        For every function it must be: function(predictions, labels) -> metric
    device : torch.device
        the device used for the training
    tag : str
        the nametag of the neural network
    best_model_dict : dict
        dictionary containing information about the best model occurred during training
    '''
    def __init__(self, model, optimizer, criterion, metric_funcs: dict, device, tag: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric_funcs = metric_funcs
        self.device = device
        self.tag = tag
        self.best_model_dict = None

    def train_single_epoch(self, dataloader):
        '''
        implements the training of the model for a single epoch

        Parameters
        ----------
        dataloader : DataLoader
            a DataLoader for the training data
            
        Returns
        -------
        tuple (float, dict)
            the training loss and a dict of training metrics 
            where each key is a string of the metrics' name 
            with value the value for that metric computed over training
        '''
        train_metrics = dict(zip(self.metric_funcs.keys(), [0.0 for _ in range(len(self.metric_funcs.keys()))]))
        train_loss = 0.0
        samples = 0

        self.model.train()
        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            y_hat = self.model(xb)
            loss = self.criterion(y_hat, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * yb.size(0)
            for key in self.metric_funcs:
                train_metrics[key] += self.metric_funcs[key](y_hat, yb) 
            samples += yb.size(0)
        
        train_loss = train_loss / samples
        for key in self.metric_funcs:
            train_metrics[key] = train_metrics[key] / samples
        return train_loss, train_metrics

    def evaluate(self, dataloader):
        '''
        evaluates the model

        Parameters
        ----------
        dataloader : DataLoader
            a DataLoader for the evaluating data
            
        Returns
        -------
        tuple (float, dict)
            tha evaluating loss and a dict of training metrics 
            where each key is a string of the metrics' name 
            with value the value for that metric computed over evaluating
        '''
        eval_metrics = dict(zip(self.metric_funcs.keys(), [0.0 for _ in range(len(self.metric_funcs.keys()))]))
        eval_loss = 0.0
        samples = 0

        self.model.eval()
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                y_hat = self.model(xb)
                loss = self.criterion(y_hat, yb)
                
                eval_loss += loss.item() * yb.size(0)
                for key in self.metric_funcs:
                    eval_metrics[key] += self.metric_funcs[key](y_hat, yb)
                samples += yb.size(0)
        
        eval_loss = eval_loss / samples
        for key in self.metric_funcs:
            eval_metrics[key] = eval_metrics[key] / samples
        return eval_loss, eval_metrics

    def fit(self, train_dataloader, val_dataloader, epochs):
        '''
        implements the training loop

        Parameters
        ----------
        train_dataloader : DataLoader
            a DataLoader for the training data
        val_dataloader : DataLoader
            a DataLoader for the validation data
        epochs: int
            the number of epochs to train the model
            
        Returns
        -------
        dict
            a dict with training and validation metrics for every epoch.
            dict['train'] and dict['validation] are dictionaries where each key is a string of the metrics' name 
            with value a tensor containing the calculated value of the metric at each epoch.
        '''
        train_loss = torch.empty(epochs)
        train_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(epochs) for _ in range(len(self.metric_funcs.keys()))]))
        val_loss = torch.empty(epochs)
        val_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(epochs) for _ in range(len(self.metric_funcs.keys()))]))

        min_val_loss = float('inf')

        for epoch in range(epochs):
            train = self.train_single_epoch(train_dataloader)
            val = self.evaluate(val_dataloader)
            #save metrics
            train_loss[epoch] = train[0]
            for key in self.metric_funcs.keys():
                train_metrics[key][epoch] = train[1][key]
            val_loss[epoch] = val[0]
            for key in self.metric_funcs.keys():
                val_metrics[key][epoch] = val[1][key]
            #save best model      
            if val_loss[epoch] < min_val_loss:
                min_val_loss = val_loss[epoch]
                self.best_model_dict = {
                        'epoch': epoch+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'training_metrics': {'loss': train[0], **train[1]},
                        'validation_metrics': {'loss': val[0], **val[1]}
                        }
        #plot learning curves                     
        plot_learning_curve(training_res=train_loss, validation_res=val_loss, metric='loss', title=self.tag, filename=f'{self.tag}_loss.png')
        for key in self.metric_funcs.keys():
            plot_learning_curve(training_res=train_metrics[key], validation_res=val_metrics[key], metric=key, title=self.tag, filename=f'{self.tag}_{key}.png')

        return {'train': {'loss': train_loss, **train_metrics}, 'validation': {'loss': val_loss, **val_metrics}}