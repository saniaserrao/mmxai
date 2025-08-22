# logger.py
import csv
import os
import matplotlib.pyplot as plt

class CSVLogger:
    def __init__(self, log_dir="results", plot_dir="plots", run_name="default_run"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.run_name = run_name
        self.log_path = os.path.join(log_dir, f"{run_name}.csv")

        # in-memory storage for plotting
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
                        "accuracy": [], "f1_weighted": [], "f1_macro": []}

        # create CSV header if not exists
        if not os.path.isfile(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "val_loss", "test_loss",
                    "accuracy", "f1_weighted", "f1_macro"
                ])
    
    def log(self, epoch, train_loss, val_loss, test_loss, acc, f1_w, f1_m):
        # update history
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["test_loss"].append(test_loss)
        self.history["accuracy"].append(acc)
        self.history["f1_weighted"].append(f1_w)
        self.history["f1_macro"].append(f1_m)

        # write to csv
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss, test_loss,
                acc, f1_w, f1_m
            ])
    
    def save_plot(self):
        # plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["epoch"], self.history["val_loss"], label="Val Loss")
        plt.plot(self.history["epoch"], self.history["test_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves - {self.run_name}")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f"{self.run_name}_loss.png"))
        plt.close()

        # plot accuracy + F1
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["epoch"], self.history["accuracy"], label="Accuracy")
        plt.plot(self.history["epoch"], self.history["f1_weighted"], label="F1 Weighted")
        plt.plot(self.history["epoch"], self.history["f1_macro"], label="F1 Macro")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"Accuracy/F1 - {self.run_name}")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f"{self.run_name}_metrics.png"))
        plt.close()
