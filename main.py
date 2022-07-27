import torch
from torch import nn
import torch.optim as optim
from data_loader import DataLoader
from itertools import product
from data import get_train_val_test, get_train_test
from sklearn.model_selection import StratifiedKFold
from metrics import binary_accuracy
from neural_network import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
EPOCHS = 100

def tune(hypers, train_data):
    min_val_loss = float("inf")
    val_performance = {hyper_set: dict() for hyper_set in hypers}

    criterion = nn.BCEWithLogitsLoss()
    metric_funcs = {'accuracy': binary_accuracy}
    for hyper_set in hypers:
        lr = hyper_set[0]
        nh = hyper_set[1]

        skf = StratifiedKFold(n_splits=5)
        for fold, (train_indices, val_indices) in enumerate(skf.split(train_data[0], train_data[1])):
            train = [train_data[0][[train_indices]], train_data[1][train_indices]]
            val = [train_data[0][val_indices], train_data[1][val_indices]]
            train_dataloader = DataLoader(train, batch_size)
            val_dataloader = DataLoader(val, batch_size)

            model = nn.Sequential(nn.Linear(len(train_data[0][0]), nh), nn.ReLU(), nn.Linear(nh, 1))
            optimizer = optim.Adam(model.parameters(), lr=lr) 
            neural_net = NeuralNetwork(criterion, optimizer, model, metric_funcs, f"{lr}_{nh}")
            cur_performance = neural_net.fit(train_dataloader, val_dataloader, EPOCHS)

            for key,value in cur_performance['validation'].items():
                if key not in val_performance[hyper_set].keys():
                    val_performance[hyper_set][key] = value
                else:
                    val_performance[hyper_set][key] = val_performance[hyper_set][key] + (1 / (fold+1)) * (value - val_performance[hyper_set][key])
            #save best model
            if neural_net.best_model_dict['validation_loss'] < min_val_loss:
                min_val_loss = neural_net.best_model_dict['validation_loss']
                best_model_dict = neural_net.best_model_dict.copy()
                best_model_dict['learning_rate'] = lr
                best_model_dict['neurons_hidden'] = nh
                torch.save(best_model_dict, 'best_model.pth')
    return val_performance


train_data, test_data = get_train_test("train.csv", "test.csv", "gender_submission.csv")

learning_rate = [0.0001, 0.001, 0.01]
neurons_hidden = [7, 4]
hypers = list(product(learning_rate, neurons_hidden))
val_performance = tune(hypers, train_data)
torch.save(val_performance, 'val_performance.pth')

#test best model
test_dataloader = DataLoader(test_data, batch_size)

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
neural_net = NeuralNetwork(criterion, optimizer, model, metric_funcs)
metrics = neural_net.evaluate(test_dataloader)
print(f'loss: {metrics[0]} metrics: {metrics[1]}')

#best cross validation model
min_loss = float('inf')
for hyper_set in hypers:
    val_performance[hyper_set]['loss'] = list(val_performance[hyper_set]['loss'])
    min_cur_loss = min(val_performance[hyper_set]['loss'])
    cur_epoch = val_performance[hyper_set]['loss'].index(min_cur_loss)
    if min_cur_loss < min_loss:
        min_loss = min_cur_loss
        best_hyper_set = hyper_set
        epoch = cur_epoch + 1
print(f'best average performance with\
        loss: {min_loss}\
        learning rate: {best_hyper_set[0]}\
        #neurons: {best_hyper_set[1]}\
        epoch: {epoch}'
    )