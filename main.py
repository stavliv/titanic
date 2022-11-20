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
EPOCHS = 2000

def tune(hypers, train_data, criterion, metric_funcs):
    min_val_loss = float("inf")
    val_performance = {hyper_set: dict() for hyper_set in hypers}

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
            neural_net = NeuralNetwork(model, optimizer, criterion, metric_funcs, device, f"{lr}_{nh}")
            cur_performance = neural_net.fit(train_dataloader, val_dataloader, EPOCHS)

            for metric,metric_tensor in cur_performance['validation'].items():
                if metric not in val_performance[hyper_set].keys():
                    val_performance[hyper_set][metric] = metric_tensor
                else:
                    val_performance[hyper_set][metric] = val_performance[hyper_set][metric] + (1 / (fold+1)) * (metric_tensor - val_performance[hyper_set][metric])
            #save best model
            if neural_net.best_model_dict['validation_metrics']['loss'] < min_val_loss:
                min_val_loss = neural_net.best_model_dict['validation_metrics']['loss']
                best_model_dict = neural_net.best_model_dict.copy()
                best_model_dict['learning_rate'] = lr
                best_model_dict['neurons_hidden'] = nh
                torch.save(best_model_dict, 'best_model.pth')
    return val_performance

def test_best_model(test_dataloader, criterion, metric_funcs):
    val_performance = torch.load('val_performance.pth')
    best_model = torch.load('best_model.pth')

    model_hyper_set = (best_model['learning_rate'], best_model['neurons_hidden'])
    val_perf_model = dict()
    for metric, metric_tensor in val_performance[model_hyper_set].items():
        val_perf_model[metric] = metric_tensor.data[best_model['epoch'] - 1].item()

    print(
        f"Best model was saved at\n\
            epochs: {best_model['epoch']} \n\
            neurons hidden: {best_model['neurons_hidden']}\n\
            learning rate: {best_model['learning_rate']}\n\n\
            ---Validation metrics---\n\
            {best_model['validation_metrics']}\n\n\
            ---Cross validation metrics---\n\
            {val_perf_model}"
        )
    model = nn.Sequential(nn.Linear(len(train_data[0][0]), best_model['neurons_hidden']), nn.ReLU(), nn.Linear(best_model['neurons_hidden'], 1))
    model.load_state_dict(best_model['model_state_dict'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_model['learning_rate'])
    neural_net = NeuralNetwork(model, optimizer, criterion, metric_funcs, device, 'best')
    metrics = neural_net.evaluate(test_dataloader)
    metrics_dict = {'loss': metrics[0], **metrics[1]}
    print(
        f'\n\
            ---Test metrics---\n\
            {metrics_dict}'
        )

def find_best_cross_model():
    val_performance = torch.load('val_performance.pth')

    min_loss = float('inf')
    for hyper_set in hypers:
        loss = list(val_performance[hyper_set]['loss'])
        min_cur_loss = min(loss)
        cur_epoch = loss.index(min_cur_loss)
        if min_cur_loss < min_loss:
            min_loss = min_cur_loss
            best_hyper_set = hyper_set
            epoch = cur_epoch + 1

    val_perf_model = dict()
    for metric, metric_tensor in val_performance[best_hyper_set].items():
        val_perf_model[metric] = metric_tensor.data[epoch - 1].item()

    print(
        f'\nBest average performance at:\n\
            learning rate: {best_hyper_set[0]}\n\
            #neurons: {best_hyper_set[1]}\n\
            epoch: {epoch}\n\n\
            ---Cross validation metrics---\n\
            {val_perf_model}'
        )


train_data, test_data = get_train_test("train.csv", "test.csv", "gender_submission.csv")
test_dataloader = DataLoader(test_data, batch_size)

learning_rate = [0.0001, 0.001, 0.01]
neurons_hidden = [7, 4]
hypers = list(product(learning_rate, neurons_hidden))

criterion = nn.BCEWithLogitsLoss()
metric_funcs = {'accuracy': binary_accuracy}

#val_performance = tune(hypers, train_data, criterion, metric_funcs)
#torch.save(val_performance, 'val_performance.pth')

test_best_model(test_dataloader, criterion, metric_funcs)
find_best_cross_model()
