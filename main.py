import torch
from torch import nn
import torch.optim as optim
from data_loader import DataLoader
from itertools import product
from data import get_train_val_test
from sklearn.model_selection import StratifiedKFold
from metrics import binary_accuracy
from neural_network import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
EPOCHS = 100

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

def tune(hypers, train_dataloader, val_dataloader):
    min_val_loss = float("inf")
    for i,hyper_set in enumerate(hypers):
            lr = hyper_set[0]
            nh = hyper_set[1]

            model = nn.Sequential(nn.Linear(len(train_data[0][0]), nh), nn.ReLU(), nn.Linear(nh, 1))
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            metric_funcs = {'accuracy': binary_accuracy}
            neural_net = NeuralNetwork(criterion, optimizer, model, metric_funcs, f"{lr}_{nh}")
            neural_net.fit(train_dataloader, val_dataloader, EPOCHS)
            #save best model
            if neural_net.best_model_dict['validation_loss'] < min_val_loss:
                min_val_loss = neural_net.best_model_dict['validation_loss']
                best_model_dict = neural_net.best_model_dict.copy()
                best_model_dict['learning_rate'] = lr
                best_model_dict['neurons_hidden'] = nh
                torch.save(best_model_dict, 'best_model.pth')


train_data, val_data, test_data = get_train_val_test("train.csv", "test.csv", "gender_submission.csv")
train_dataloader = DataLoader(train_data, batch_size)
val_dataloader = DataLoader(val_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

learning_rate = [0.0001, 0.001, 0.01]
neurons_hidden = [7, 4]
hypers = product(learning_rate, neurons_hidden)
#tune(hypers, train-dataloader, val_dataloader)
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