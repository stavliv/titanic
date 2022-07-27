import torch
from plot import plot_learning_curve

class NeuralNetwork():
    def __init__(self, criterion, optimizer, model, metric_funcs, device, tag):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.metric_funcs = metric_funcs
        self.device = device
        self.tag = tag
        self.best_model_dict = None

    def train(self, dataloader):
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
        return {'loss': eval_loss, **eval_metrics}

    def fit(self, train_dataloader, val_dataloader, epochs):
        train_loss = torch.empty(epochs)
        train_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(epochs) for _ in range(len(self.metric_funcs.keys()))]))
        val_loss = torch.empty(epochs)
        val_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(epochs) for _ in range(len(self.metric_funcs.keys()))]))

        min_val_loss = float('inf')

        for epoch in range(epochs):
            train = self.train(train_dataloader)
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