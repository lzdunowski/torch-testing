import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from models.simple_cnn import SimpleCNN
from utils import imshow
import matplotlib.pyplot as plt

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
    
    # 4. splitting the set to get train+validation
    train_size = 45000
    val_size = 5000
    trainset, valset = random_split(trainset, [train_size, val_size])

    # 5. Dataloaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader   = DataLoader(valset, batch_size=64, shuffle=False)

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
    
    # Training with validation
    num_epochs = 20
    best_val_acc = 0.0 #to store best outcome
    
    for epoch in range(num_epochs):  # swapped to switchable amount
        model.train()
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
            
        train_loss = running_loss / len(trainloader)
        
        #VALIDATION
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(valloader)
        val_acc = 100 * correct / total

        print(f"Epoka {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_acc:.2f}%")

        # Saving the best model after training
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved - saved to best_model.pth")
        
    #Saving the last state
    torch.save(model.state_dict(), "model_last.pth")
    print("Training finished! ")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
        
        
   

if __name__ == "__main__":
    main()
