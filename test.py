import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from models.simple_cnn import SimpleCNN
from utils import imshow
import torchvision

def main():
    # Data tranformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load testing data
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Read model
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Displaying sample images with predictions
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    for i in range(5):
        imshow(torchvision.utils.make_grid(images))

    print("Rzeczywiste etykiety(test):", labels.numpy())
    print("Przewidywane etykiety(test):", predicted.numpy())

    # Model testing
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Dokładność: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
