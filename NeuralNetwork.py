import torch
from torch import nn
import torch.optim as optim
from DataLoader import DataLoader
from itertools import product
from Data import get_data
from plot import plot_learning_curves

batch_size = 32
EPOCHS = 100

class NeuralNetwork():
    def __init__(self, hypers, train_data, val_data, test_data):
        self.hypers = hypers

        self.train_dataloader = DataLoader(train_data, batch_size)
        self.val_dataloader = DataLoader(val_data, batch_size)
        self.test_dataloader = DataLoader(test_data, batch_size)

        self.input_size = len(train_data[0][0]) 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        performance = dict()
        min_val_loss = float("inf")

        for i,hyper_set in enumerate(self.hypers):
            lr = hyper_set[0]
            nh = hyper_set[1]

            model = nn.Sequential(nn.Linear(self.input_size, nh), nn.ReLU(), nn.Linear(nh, 1))
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            epoch_train_loss = list()
            epoch_train_accuracy = list()
            epoch_val_loss = list()
            epoch_val_accuracy = list()

            for epoch in range(EPOCHS):
                print(f"hypers: {hyper_set} epoch: {epoch + 1}")
                #training
                train_acc = 0.0
                train_loss = 0.0
                samples = 0

                model.train()
                for xb, yb in self.train_dataloader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    
                    y_hat = model(xb)
                    loss = criterion(y_hat, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * yb.size(0)
                    train_acc += (torch.where(y_hat >= 0.5, 1., 0.) == yb).sum().item()
                    samples += yb.size(0)
                
                train_loss = train_loss / samples
                train_acc = train_acc / samples
                epoch_train_loss.append(train_loss)
                epoch_train_accuracy.append(train_acc)

                #validation
                val_acc = 0.0
                val_loss = 0.0
                samples = 0

                model.eval()
                with torch.no_grad():
                    for xb, yb in self.val_dataloader:
                        xb, yb = xb.to(self.device), yb.to(self.device)

                        y_hat = model(xb)
                        loss = criterion(y_hat, yb)
                        
                        val_loss += loss.item() * yb.size(0)
                        val_acc += (torch.where(y_hat >= 0.5, 1., 0.) == yb).sum().item()
                        samples += yb.size(0)
                
                val_loss = val_loss / samples
                val_acc = val_acc / samples
                epoch_val_loss.append(val_loss)
                epoch_val_accuracy.append(val_acc)

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(
                        {
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'learning_rate': lr,
                        'neurons_hidden': nh,
                        'training_loss': train_loss,
                        'training_accuracy': train_acc,
                        'validation_loss': val_loss,
                        'validation_accuracy': val_acc
                        }, 
                        'best_model.pth')

            performance[hyper_set] = (
                {
                "loss": {"training": epoch_train_loss, "validation": epoch_val_loss}, 
                "accuracy": {"training": epoch_train_accuracy, "validation": epoch_val_accuracy}
                })

        plot_learning_curves(performance)

    def evaluate(self):
        best_model = torch.load('best_model.pth')
        best_model_epoch = best_model['epoch']
        print(f"Best model was saved at\n\
                epochs: {best_model_epoch} \n\
                neurons hidden: {best_model['neurons_hidden']}\n\
                learning rate: {best_model['learning_rate']}"
            )
        model = nn.Sequential(nn.Linear(self.input_size, best_model['neurons_hidden']), nn.ReLU(), nn.Linear(best_model['neurons_hidden'], 1))
        model = model.to(self.device)
        model.load_state_dict(best_model['model_state_dict'])

        criterion = nn.BCEWithLogitsLoss()

        test_acc = 0.0
        test_loss = 0.0
        samples = 0

        model.eval()
        with torch.no_grad():
            for xb, yb in self.test_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                y_hat = model(xb)
                loss = criterion(y_hat, yb)
                
                test_loss += loss.item() * yb.size(0)
                test_acc += (torch.where(y_hat >= 0.5, 1., 0.) == yb).sum().item()
                samples += yb.size(0)
        
        test_loss = test_loss / samples
        test_acc = test_acc / samples

        print(f"loss: {test_loss} accuracy: {test_acc}")

if __name__ == '__main__':
    train_data, val_data, test_data = get_data("train.csv", "test.csv", "gender_submission.csv")    

    learning_rate = [0.0001, 0.001, 0.01]
    neurons_hidden = [7, 4]
    hypers = product(learning_rate, neurons_hidden)

    neural_net = NeuralNetwork(hypers, train_data, val_data, test_data)
    #neural_net.train()
    neural_net.evaluate()