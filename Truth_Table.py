#CRIAR UMA REDE NEURAL QUE PREENCHA A TABELA AND

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#1º PASSO CRIAR O DATASET

class ANDDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ], dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, -1]
        return x, y

dataset = ANDDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class ANDNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4,1),#2 inputs 1 output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ANDNetwork().to(device)
lossfunc = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train(model, dataloader, lossfunc, optimizer):
    model.train()  # Configurar o modelo para treinamento
    cumloss = 0.0
    for x, y in dataloader:  # Iterar sobre o dataloader para obter os lotes
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(1)
        pred = model(x)
        loss = lossfunc(pred, y)

        optimizer.zero_grad()  # Zerar os gradientes
        loss.backward()        # Calcular os gradientes
        optimizer.step()       # Atualizar os pesos

        cumloss += loss.item()  # Acumular a perda
    return cumloss / len(dataloader)

def test(model, dataset, lossfunc):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in dataset:
            x = x.unsqueeze(0).to(device)
            pred = model(x).round().item()  # Arredonda a saída e converte para valor numérico
            predictions.append((x.cpu().squeeze().tolist(), pred))  # Armazena entradas e predições
    return predictions

epochs = 500

for t in range(epochs):
    train_loss = train(model, dataloader, lossfunc, optimizer)
    if t % 100 == 0:
        print(f"Epoch: {t}, Train Loss: {train_loss: 4f}")
    print(f"Predição: {train_loss}")

predictions = test(model, dataset, lossfunc)
print("\True Table AND:")
for inputs, pred in predictions:
    print(f"Entradas: {inputs}, Predicao: {pred}")


