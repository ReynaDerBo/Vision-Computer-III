import matplotlib.pyplot as plt

def plot_history(history, output_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SmartEye - Accuracy")
    plt.legend()
    plt.grid()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
