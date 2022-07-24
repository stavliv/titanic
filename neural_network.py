import torch
from torch import nn
import torch.optim as optim
from data_loader import DataLoader
from itertools import product
from data import get_train_val_test
from plot import plot_learning_curve, plot_learning_curves
from sklearn.model_selection import StratifiedKFold
from metrics import binary_accuracy

batch_size = 32
EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def k_fold(train_data):
    train_data[0] = torch.tensor(train_data[0])
    train_data[1] = torch.tensor(train_data[1])
    skf = StratifiedKFold(n_splits=5)
    (train_indices, val_indices) = skf.split(train_data[0], train_data[1])
    train = [train_data[0][[train_indices]], train_data[1][train_indices]]
    val = [train_data[0][val_indices], train_data[1][val_indices]]
    train_dataloader = DataLoader(train, batch_size)
    val_dataloader = DataLoader(val, batch_size)
    return train_dataloader, val_dataloader

def tune(hypers):
    train_data, val_data, test_data = get_train_val_test("train.csv", "test.csv", "gender_submission.csv")
    
    train_dataloader = DataLoader(train_data, batch_size)
    val_dataloader = DataLoader(val_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)

    min_val_loss = float("inf")
    for i,hyper_set in enumerate(hypers):
            lr = hyper_set[0]
            nh = hyper_set[1]

            model = nn.Sequential(nn.Linear(len(train_data[0][0]), nh), nn.ReLU(), nn.Linear(nh, 1))
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            metric_funcs = {'accuracy': binary_accuracy}
            neural_net = NeuralNetwork(criterion, optimizer, model, metric_funcs, f"{lr}_{nh}")
            neural_net.fit(train_dataloader, val_dataloader)
            #save best model
            if neural_net.best_model_dict['validation_loss'] < min_val_loss:
                min_val_loss = neural_net.best_model_dict['validation_loss']
                best_model_dict = neural_net.best_model_dict.copy()
                best_model_dict['learning_rate'] = lr
                best_model_dict['neurons_hidden'] = nh
                torch.save(best_model_dict, 'best_model.pth')


class NeuralNetwork():
    def __init__(self, criterion, optimizer, model, metric_funcs, tag):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.metric_funcs = metric_funcs
        self.tag = tag
        self.best_model_dict = None
        self.input_size = len(train_data[0][0]) 

    def train(self, dataloader):
        train_metrics = dict(zip(self.metric_funcs.keys(), [0.0 for _ in range(len(self.metric_funcs.keys()))]))
        train_loss = 0.0
        samples = 0

        self.model.train()
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
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
                xb, yb = xb.to(device), yb.to(device)

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

    def fit(self, train_dataloader, val_dataloader):
        train_loss = torch.empty(EPOCHS)
        train_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(EPOCHS) for _ in range(len(self.metric_funcs.keys()))]))
        val_loss = torch.empty(EPOCHS)
        val_metrics = dict(zip(self.metric_funcs.keys(), [torch.empty(EPOCHS) for _ in range(len(self.metric_funcs.keys()))]))

        min_val_loss = float('inf')

        for epoch in range(EPOCHS):
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
                        'training_loss': train[0],
                        'training_metrics': train[1],
                        'validation_loss': val[0],
                        'validation_metrics': val[1]
                        }
        #plot learning curves                     
        plot_learning_curve(training_res=train_loss, validation_res=val_loss, metric='loss', title=self.tag, filename=f'{self.tag}_loss.png')
        for key in self.metric_funcs.keys():
            plot_learning_curve(training_res=train_metrics[key], validation_res=val_metrics[key], metric=key, title=self.tag, filename=f'{self.tag}_{key}.png')

        return train_loss, train_metrics

if __name__ == '__main__':   
    train_data, val_data, test_data = get_train_val_test("train.csv", "test.csv", "gender_submission.csv")
    train_dataloader = DataLoader(train_data, batch_size)
    val_dataloader = DataLoader(val_data, batch_size)
    test_dataloader = DataLoader(test_data, batch_size)
    
    learning_rate = [0.0001, 0.001, 0.01]
    neurons_hidden = [7, 4]
    hypers = product(learning_rate, neurons_hidden)
    #tune(hypers)
    #test best model
    best_model = torch.load('best_model.pth')
    best_model_epoch = best_model['epoch']
    print(f"Best model was saved at\n\
            epochs: {best_model_epoch} \n\
            neurons hidden: {best_model['neurons_hidden']}\n\
            learning rate: {best_model['learning_rate']}"
        )
    model = nn.Sequential(nn.Linear(len(train_data[0][0]), best_model['neurons_hidden']), nn.ReLU(), nn.Linear(best_model['neurons_hidden'], 1))
    model.load_state_dict(best_model['model_state_dict'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_model['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    metric_funcs = {'accuracy': binary_accuracy}
    neural_net = NeuralNetwork(criterion, optimizer, model, metric_funcs, train_data, val_data, test_data)
    metrics = neural_net.evaluate(test_dataloader)
    print(f'loss: {metrics[0]} metrics: {metrics[1]}')
