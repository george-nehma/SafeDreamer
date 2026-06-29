from info_nce import InfoNCE, info_nce
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NPZDataset(Dataset):
    def __init__(self,directory):
        self.files = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npz")]
        )

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        state = torch.tensor(data["state"], dtype=torch.float32)
        return state

    def __len__(self):
        return len(self.files)
    


class SafetyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SafetyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return F.log_softmax(out, dim=-1)
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = SafetyClassifier(input_size=14, hidden_size=100, output_size=1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    num_epochs = 50


    dataset = NPZDataset("/workspace/SafeDreamer/logs/lander_6dof_direct/20260217_164140/train_eps/")
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)



    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: process input through the network
            outputs = classifier(inputs)

            # Compute loss (negative log-likelihood)
            loss = criterion(outputs, labels)

            # Backward pass: propagate gradients back into the network parameters
            loss.backward()

            # Update the weights
            optimizer.step()





# loss = InfoNCE()
# batch_size, embedding_size = 32, 128
# query = torch.randn(batch_size, embedding_size)
# positive_key = torch.randn(batch_size, embedding_size)
# output = loss(query, positive_key)

# print(f"""InfoNCE Loss: {output.item()}
#           Query:     {query}
#           Positive Key: {positive_key}""")