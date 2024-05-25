import pickle
import numpy as np
import sklearn as sk
import torch
from torch import nn, cuda
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

device = "cuda" if cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def convert_to_tensors(data):
    return torch.from_numpy(np.asarray(data))

def begin_plot():
    plt.figure(figsize=(15,8))

def finish_plot():
    plt.gca().set_ylim([0, 100])
    plt.legend()
    plt.show()

class GenericBinaryClassifier(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
    def save_model(self, output_path):
        torch.save(self.state_dict(), f=output_path)
    
    def load_model(self, input_path):
        self.load_state_dict(torch.load(input_path)) 
    
    def train_model(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=25000, 
            learning_rate=0.05, 
            should_print=False,
            should_plot=False,
            plot_label=None,
            analysis_interval=100,
            print_interval=500,
            ):
        
        # make the results repeatable
        torch.manual_seed = 12345

        # define loss function and optimizer function
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # save data to plot later
        epoch_data = []
        val_accuracy = []

        # run the training loop
        for epoch in range(1, epochs+1):
            self.train()

            # train model
            y_logits = self(X_train).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_train, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log training stats
            if epoch % analysis_interval == 0:
                self.eval()
                with torch.inference_mode():
                    val_logits = self(X_val).squeeze()
                    val_pred = torch.round(torch.sigmoid(val_logits))
                    val_acc = accuracy_fn(y_val, val_pred)
                    if epoch % print_interval == 0 and should_print:
                        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%")
                
                    # plot data
                    if should_plot:
                        epoch_data.append(epoch)
                        val_accuracy.append(val_acc)


        if should_plot:
            plt.plot(epoch_data, val_accuracy, label=plot_label)

        # generate final classification report
        self.eval()
        with torch.inference_mode():
            val_logits = self(X_val).squeeze()
            val_pred = torch.round(torch.sigmoid(val_logits))
            report = classification_report(y_val.cpu(), val_pred.cpu())

        return report
    
class GenericMutliClassifier(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8, report_labels=None):
        super().__init__()
        self.report_labels = report_labels
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
    def save_model(self, output_path):
        torch.save(self.state_dict(), f=output_path)
    
    def load_model(self, input_path):
        self.load_state_dict(torch.load(input_path)) 
    
    def train_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=25000, 
        learning_rate=0.05, 
        should_print=False,
        should_plot=False,
        plot_label=None,
        analysis_interval=100,
        print_interval=500):

        # make results reproducable with a seed
        torch.manual_seed = 12345
        torch.cuda.manual_seed(12345)
        np.random.seed(12345)

        # define loss function and optimiser
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # save data to plot later
        epoch_data = []
        val_accuracy = []

        for epoch in range(1, epochs+1):

            # train model
            self.train()
            y_logits = self(X_train)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_train, y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % analysis_interval == 0:
                self.eval()
                with torch.inference_mode():
                    val_logits = self(X_val)
                    val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
                    val_acc = accuracy_fn(y_val, val_pred)
                    if epoch % print_interval == 0 and should_print:
                        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%")
                
                    # plot data
                    if should_plot:
                        epoch_data.append(epoch)
                        val_accuracy.append(val_acc)

        if should_plot:
            plt.plot(epoch_data, val_accuracy, label=plot_label)

        # generate final classification report
        self.eval()
        with torch.inference_mode():
            val_logits = self(X_val)
            val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
            report = classification_report(y_val.cpu(), val_pred.cpu(), target_names=self.report_labels)

        return report

    def evaluate_model(
            self, 
            X_val,
            ):
        self.eval()
        with torch.inference_mode():
            val_logits = self(X_val)
            val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
        return val_pred