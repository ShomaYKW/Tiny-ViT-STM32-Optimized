import matplotlib.pyplot as plt

def plot_training_history(train_accs, val_accs, train_losses, val_losses):
    
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(12, 5))

    # plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, 'b-', label='Training Acc')
    plt.plot(epochs, val_accs, 'r-', label='Validation Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot 
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Plot saved ")
    plt.close() 