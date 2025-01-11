import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from models.simple_cnn import SimpleCNN
from utils import imshow

def main():
    # setting up GPU as default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Data transformation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Displaying sample images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print("Actual labels (train):", labels.numpy())
    imshow(torchvision.utils.make_grid(images))

    # model init
    print("------init model  ------")
    model = SimpleCNN().to(device) #transfering to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("------init model - done ------")
    # Train
    for epoch in range(20):  # added to 20 - testing to see if model would overfit
        running_loss = 0.0
        for inputs, labels in trainloader:
            #transfering to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    # Saving model after training
    torch.save(model.state_dict(), "model.pth")
    print("Model put in model.pth")

if __name__ == "__main__":
    main()
