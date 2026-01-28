import torch

def debug_single_batch(model, train_loader, device):

    model.to(device)
    model.train()
    
    # 1. Get just ONE batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # Strong LR
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(100): # 100 steps
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels).sum().item() / labels.size(0) * 100
            print(f"Step {i+1}: Loss {loss.item():.4f} | Accuracy: {acc:.0f}%")
            
            if acc == 100:
                print("SUCCESS: Model can learn! The issue is hyperparameters.")
                return

    print("FAILURE: Model cannot overfit. The code/architecture is broken.")

