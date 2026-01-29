import torch 
import torch.nn as nn


def run_inference(test_loader, model, num_epoch, device):
    
    test_acc_list = [] 

    criterion = nn.CrossEntropyLoss()
    #set the model to evaluation mode
    model.eval()
    #cumulative loss, number of correct predictions, and total number of samples processed
    running_loss, correct, total = 0.0, 0, 0

    for epoch in range(num_epoch):
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

        test_loss, test_acc = running_loss / len(test_loader), 100. * correct / total
        test_acc_list.append(test_acc)

        print(f"Epoch: {epoch+1}/{num_epoch} | Test Acc: {test_acc:.2f}%")
        avg_accuracy = sum(test_acc_list) // len(test_acc_list)

    print(f"average test accuracy: {avg_accuracy}")     
    return avg_accuracy   