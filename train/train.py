import torch
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, criterion, optimizer, device):

    #set the model to training mode
    model.train()

    #cumulative loss, number of correct predictions, and total number of samples processed
    running_loss, correct, total = 0.0, 0, 0

   #for every input and label in dataset
    for inputs, labels in train_loader:

         #moves the input data and labels to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        #optimizer for the gredient : important because python accumlates the gredient by default
        optimizer.zero_grad()
        #forward pass 
        outputs = model(inputs)
        #loss calculation
        loss = criterion(outputs, labels)
        #backward pass
        loss.backward()
        #update the weight
        optimizer.step()

        #adding the parameters up after iteration for tracking purposes

        #add the current batch's loss to the running total
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        #add the current batch size to the total count of processed samples
        total += labels.size(0)
        #compare predictions with labels, sums the correct ones and adds to the count
        correct += predicted.eq(labels).sum().item()

#return the average loss per batch and the accuracy 
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):

    #set the model to evaluation mode
    model.eval()
    #cumulative loss, number of correct predictions, and total number of samples processed
    running_loss, correct, total = 0.0, 0, 0

    #disable gredient calculation for faster evaluation
    with torch.no_grad():
            for inputs, labels in test_loader:
                 inputs, labels = inputs.to(device), labels.to(device)

                 #output by the model
                 outputs = model(inputs)
                 #calculate the difference between label and the output by the model (loss)
                 loss = criterion ( outputs, labels)
                 running_loss += loss.item()

                 _, predicted = outputs.max(1)
                 total += labels.size(0)
                 correct += predicted.eq(labels).sum().item()

    return running_loss / len(test_loader), 100. * correct / total

def run_training(train_loader, test_loader, model, num_epochs, lr ):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay= 1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
          
        print(f"Epoch: {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")
    
    return model

        