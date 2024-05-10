import matplotlib.pyplot as plt
import numpy as np
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch import nn
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import optuna
from helper_functions import accuracy_fn
from timeit import default_timer as timer
import torch

torch.cuda.manual_seed(42)
torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self, input_size, n_neurons, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_neurons)
        self.out = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.out(x)
        return x


transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

print(f"Number of images in train set is {len(train_set)}")
print(f"Number of images in test set is {len(test_set)}")

digits = set()
fig, axes = plt.subplots(1, 10, figsize=(10, 2))

for images, labels in train_loader:
    for image, label in zip(images, labels):
        if label.item() not in digits:
            digits.add(label.item())
            ax = axes[label.item()]
            ax.imshow(image.squeeze(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Label: {label.item()}')

        if len(digits) == 10:
            break
    if len(digits) == 10:
        break

plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Hyperparameter Tuning time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X = X.view(X.shape[0], -1)
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X = X.view(X.shape[0], -1)
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)
                                    )

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_acc


def objective(trial):
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    train_set = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    n_neurons = trial.suggest_categorical('n_neurons', [32, 64, 128])
    model = MLP(784, n_neurons, 10)
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    lr = trial.suggest_categorical('lr', [0.001, 0.01, 0.1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 25
    accuracy_array = np.array([])
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_loader,
                   model=model,
                   loss_fn=criterion,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn
                   )
        test_acc = test_step(data_loader=test_loader,
                             model=model,
                             loss_fn=criterion,
                             accuracy_fn=accuracy_fn
                             )
        accuracy_array = np.append(accuracy_array, test_acc)
    return np.mean(accuracy_array[-10:])


train_time_start_on_gpu = timer()
study = optuna.create_study(direction='maximize',
                            storage="sqlite:///db_TPE.sqlite3",
                            study_name="MNIST-MLP-Classification-TPE",
                            pruner=MedianPruner(),
                            sampler=TPESampler()
                            )
study.optimize(objective, n_trials=30)
train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


