"""
Implements cnn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random 
from tqdm import tqdm
from torch.utils.data import Dataset


def dataset_preprocessing(dataset):
  train_indices = [i for i, (_, label) in enumerate(dataset) if label in [0, 1]]
  trainset = torch.utils.data.Subset(dataset, train_indices)
  return trainset


def trainTest_model(model, epoch_num, train_loader, testloader, optimizer, criterion, device):
  loss_epoch = []
  acc_epoch = []
  test_acc = []
  for epoch in range(epoch_num):  
    running_loss = 0.0
    acc = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for data in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc  += (100 * correct / total)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

      loss_epoch.append(running_loss/(len(train_loader)))
      acc_epoch.append(acc/(len(train_loader))) 
      test_acc.append(testModel(model, testloader, device))





  return model, loss_epoch, acc_epoch, test_acc




def testModel(model, testloader, device):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return (100 * correct / total)  



  # part b) custom dataset class:



# Define the Dataset class for triplet loss
class customDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = len(dataset[:][1])
        self.classes = dataset[:][1]
        self.targets = dataset[:][1]

    def __getitem__(self, index):
        img1, lbl1 = self.dataset[index]
        positive_indices = [i for i, lbl in enumerate(self.dataset[:][1]) if lbl == lbl1]
        negative_indices = [i for i, lbl in enumerate(self.dataset[:][1]) if lbl != lbl1]
        img2, lbl2 = self.dataset[positive_indices[random.randint(0, len(positive_indices) - 1)]]
        img3, lbl3 = self.dataset[negative_indices[random.randint(0, len(negative_indices) - 1)]]

        return img1, lbl1, img2, lbl2, img3, lbl3

    def __len__(self):
        return len(self.dataset)


def train_model_triplet(model,train_loader,num_epochs,device,optimizer,criterion):
  total_step = len(train_loader)
  epoch_loss = []
  for epoch in range(num_epochs):
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for (anchor, lbl1, positive, lbl2, negative, lbl3) in tepoch:
          tepoch.set_description(f"Epoch {epoch}")
          anchor = anchor.to(device)
          positive = positive.to(device)
          negative = negative.to(device)
          # Forward pass
          anchor_output = model(anchor)
          positive_output = model(positive)
          negative_output = model(negative)

          # Compute triplet loss
          loss = criterion(anchor_output, positive_output, negative_output)
          running_loss += loss.item()
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          tepoch.set_postfix(loss=loss.item())
      epoch_loss.append(running_loss/len(train_loader))

  return model



def train_classifier_triplet(model, train_loader,test_loader, optimizer,criterion,device,num_epochs):
  loss_epoch = []
  acc_epoch = []
  test_acc = []
  for epoch in range(5):  # Adjust the number of epochs as needed
    acc = 0.0
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for (anchor, lbl1, positive, lbl2, negative, lbl3) in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        inputs, labels = anchor,lbl1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc  += (100 * correct / total)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tepoch.set_postfix(loss=loss.item(), accuracy=(100 * correct / total))
      acc_epoch.append(acc/len(train_loader))
      loss_epoch.append(running_loss/len(train_loader))
      acc = testModel_triplet(model, test_loader, device)
      test_acc.append(acc)  
  return model, loss_epoch, acc_epoch, test_acc

def testModel_triplet(model, test_loader, device):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return (100 * correct / total)


# Part 3:

def train_model_triplet2(model,num_epochs,train_loader,test_loader,device,criterion_1,criterion_2,optimizer):
  acc_epoch = []
  loss_epoch = []
  test_acc = []
  for epoch in range(num_epochs):
    acc = 0.0
    running_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
      for (anchor, lbl1, positive, lbl2, negative, lbl3) in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        anchor = anchor.to(device)
        lbl1 = lbl1.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        # Forward pass
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        # Compute triplet loss
        loss = criterion_1(anchor_output, positive_output, negative_output) + criterion_2(anchor_output, lbl1)
        _, predicted = torch.max(positive_output.data, 1)
        total = lbl1.size(0)
        correct = (predicted == lbl1).sum().item()
        acc  += (100 * correct / total)
        running_loss += loss.item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tepoch.set_postfix(loss=loss, accuracy=(100 * correct / total))
      acc_epoch.append(acc/len(train_loader))
      loss_epoch.append(running_loss/len(train_loader))
      acc = testModel_triplet(model, test_loader, device)
      test_acc.append(acc)
  return model, loss_epoch, acc_epoch, test_acc







