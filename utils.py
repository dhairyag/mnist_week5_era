# utils.py
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Train data transformations
# define transformations for pre-processing training and
#  test datasets before feeding them into a neural network
# compose:  take a list of transformations as input and apply them sequentially to the input data
train_transforms = transforms.Compose([
    # 10% chance that the center crop transformation will be applied to any given image
    # crop the given image at the center to have a final size of 22x22 pixels
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    #  ensure that all images have the same size before being fed into a neural network
    transforms.Resize((28, 28)),
    # Randomly rotates the image by a degree selected uniformly in -15 to 15
    # increase the robustness of the model to variations in the orientation
    transforms.RandomRotation((-15., 15.), fill=0),
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.ToTensor(),
    # Normalizes a tensor image with mean and standard deviation
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
# Similar to the training transforms, it chains the transformations for the test data.
test_transforms = transforms.Compose([
    #transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    #transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    #  ensure that all images have the same size before being fed into a neural network
    #transforms.Resize((28, 28)),
    #transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    #transforms.Normalize((0.1407,), (0.4081,))
    transforms.Normalize((0.1307,), (0.3081,))
    ])


def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

