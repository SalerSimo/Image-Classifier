from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class ConvNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(32*32*32, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))

        x = x.view(-1, 32*32*32)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def LoadData(image_height: int, image_width: int, batchSize: int, directory: str) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=directory, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True)

    return train_loader, test_loader


def TrainModel(train_loader: DataLoader, test_loader: DataLoader, verbose: bool = False) -> ConvNeuralNetwork:
    model = ConvNeuralNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if(verbose):
                if i % 10 == 9:
                    print(f"\tEpoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1} \tAccuracy: {100*correct/total}%")
    return model

def SaveModel(model: ConvNeuralNetwork, model_path: str) -> None:
    torch.save(model.state_dict(), model_path)

def LoadModel(model_path: str) -> ConvNeuralNetwork:
    model = ConvNeuralNetwork()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def MakePrediction(model: ConvNeuralNetwork, image_path: str, class_names: list[str], image_size: tuple[int, int]) -> str:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return class_names[predicted_class.item()]