import os
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/main/pytorchtools.py
    """
    def __init__(self, patience=10, delta=0, path='checkpoint.pth', trace_func=print, verbose=False):
        self.patience = patience
        # When validation loss fluctuates greatly, a larger delta should be set(1e-3)
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model = None
        self.best_epoch = -1

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace(f"ES Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """Saves model when validation loss decreases."""
        dir = os.path.dirname(self.path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        if self.verbose:
            self.trace(f"Val loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...")
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch

    def trace(self, msg):
        if self.trace_func is not None:
            self.trace_func(msg)

if __name__ == '__main__':
    pass
