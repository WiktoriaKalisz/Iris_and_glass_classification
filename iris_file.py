# Zaimportowanie odpowiednich bibliotek
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


num_words = 4

# TODO Zaladowanie danych z pliku csv do tabicy numpy

data_frame = pd.read_csv("Iris.csv")

sns.set()
sns.pairplot(data_frame[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],
             hue="Species", diag_kind="kde", height=1)
plt.show()

# TODO Podział danych na x i y

x_arr = data_frame.drop(['Id', 'Species'], axis=1)
y_first = data_frame['Species']

print(x_arr.shape)
# print(y.head())
print(y_first.shape)

# TODO Podmiana etykiet w wektorze y
y_vec=np.zeros((len(y_first),3))

for i in range(len(y_first)):
        if y_first[i]=='Iris-setosa':
            y_vec[i][0]=1.0
        elif y_first[i]=='Iris-versicolor':
            y_vec[i][1]=1.0
        elif y_first[i]=='Iris-virginica':
            y_vec[i][2]=1.0

y_vec=np.float64(y_vec)


# TODO Podział danych na zbiór treningowy i testowy z użyciem sklearn.split_costam

X_train, X_test, y_train, y_test = train_test_split(x_arr, y_vec, test_size=0.3, random_state=70) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# TODO Normalizacja argumentów x (train i test)

x_min_column_wise = X_train.min(axis=0)
x_max_column_wise = X_train.max(axis=0)
x_max_column_wise[x_max_column_wise == 0] = 1
X_train_normalized = (X_train - x_min_column_wise) / (x_max_column_wise - x_min_column_wise)

X_test_normalized = (X_test - x_min_column_wise) / (x_max_column_wise - x_min_column_wise)



# TODO Utowrzenie tensorów ze zioru testowego x i y oraz trenigowego x i y
X_train_normalized=X_train_normalized.to_numpy()
X_test_normalized=X_test_normalized.to_numpy()



X_train_normalized = torch.from_numpy(X_train_normalized)
X_test_normalized = torch.from_numpy(X_test_normalized)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)




# TODO Utworzenie dataloadera dla zbioru treningowego

dataset_train = TensorDataset(X_train_normalized, y_train)
trainloader = DataLoader(dataset_train, batch_size=10, shuffle=True)

# TODO Utworzenie instancji sieci neuronowej (np polecenie Sequential lub jako klasa)
net = torch.nn.Sequential(
    torch.nn.Linear(num_words, 3),
    torch.nn.Tanh(),
    torch.nn.Linear(3, 3),
    torch.nn.Tanh(),
    torch.nn.Linear(3, 3),
    torch.nn.Softmax()
).double()


# TODO Wybór i pobranie optimizera dla sieci

learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

# TODO Szerokopojęty trening sieic neuronowej przez n epok
nb_epochs = 700

loss = torch.nn.MSELoss()

train_losses = []
test_losses = []

for epoch in range(nb_epochs):

    for x, y in trainloader:

        output = net(x)

        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    train_loss = loss(net(X_train_normalized), y_train).item()
    test_loss = loss(net(X_test_normalized), y_test).item()
    print(f"epoch {epoch + 1} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")
    train_losses.append(train_loss)
    test_losses.append(test_loss)


# TODO Wyświetlenie funkcji strat od epoki dla zbioru testowego i treningowego

# Parametr bo definiuje linię przerywaną w postaci niebieskich kropek.
plt.plot(train_losses, label='Strata trenowania')
# Parametr b definiuje ciągłą niebieską linię.
plt.plot(test_losses, label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.show()

predicted_labels = np.argmax(net(X_test_normalized).detach(), axis=1)
accuracy = sum((predicted_labels) == np.argmax(y_test.detach(), axis=1)).double().item() / len(predicted_labels)
print(f"accuracy: {accuracy:.2f}")
report = skl.metrics.classification_report(np.argmax(y_test.detach(), axis=1), predicted_labels)
print(report)

sns.heatmap(confusion_matrix(np.argmax(y_test.detach(), axis=1), predicted_labels), annot=True, cmap = 'viridis', fmt='.0f')
plt.show()