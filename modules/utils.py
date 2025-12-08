import random
import copy
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm.auto import tqdm # for progress bars

from modules.medium_mlp import ObjectClassifier # model

class Utils:
    @staticmethod
    def set_seed(seed):
        """
        Sets seed for reproducibility.

        Args:
            seed (int): random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def save_checkpoint(self, model, params, path, train_accs=None, val_accs=None, losses=None, full_train_dataset=False):
        """
        Save model weights + model parameters (hyperparameters).

        Args:
            model: the model instance
            params: dict of hyperparameters (e.g., hidden_size, dropout, etc.)
            path: file path for saving (do not hard code this for modularity)
            train_accs: training accuracy
            val_accs: validation accuracy
            losses: training loss
            full_train_dataset: whether the model was trained on the full train dataset or not
        """
        if not full_train_dataset:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_params": params,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "losses": losses,
            }
        else:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_params": params,
            }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path, device="cpu", full_train_dataset=False):
        """
        Load a model checkpoint including architecture parameters.

        Args:
            model_class: the class of the model (e.g., MyMLP)
            path: path to the checkpoint file
            device: cpu or cuda
            full_train_dataset: whether the model was trained on the full train dataset or not

        Returns:
            model: instantiated model with weights loaded
            params: the saved model parameters
        """
        checkpoint = torch.load(path, map_location=device)
        params = checkpoint["model_params"]

        # instantiate the model using saved params
        model = ObjectClassifier(hidden_size=params["hidden_size"], output_size=50, dropout=params["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        if not full_train_dataset:
            train_accs = checkpoint["train_accs"]
            val_accs = checkpoint["val_accs"]
            losses = checkpoint["losses"]
            return model, params, train_accs, val_accs, losses
        else:
            return model, params



    def train_and_validate(
            self,
            train_tensor,
            val_tensor,
            device,
            params,
            grad_clip=True,
            gauss=True,
            log=True,
            patience=5,
    ):
        """
        Train the model on a training dataset and evaluate on a validation dataset.

        Args:
            train_tensor: training data or DataLoader containing features and labels
            val_tensor: validation data or DataLoader containing features and labels
            device: device to run training on ("cpu" or "cuda")
            params: dictionary of training hyperparameters
            grad_clip: whether to apply gradient clipping during training
            gauss: whether to apply Gaussian noise
            log: whether to log training and validation metrics during training
            patience: number of epochs to wait for validation improvement before early stopping

        Returns:
            model: the trained model with the best validation weights restored
            train_accs: training accuracy
            val_accs: validation accuracy
            losses: training loss
            best_val_accs: best validation accuracy
        """
        best_val_acc = 0
        train_accs = []
        val_accs = []
        losses = []
        best_model_wts = None
        epochs_no_improve = 0

        self.set_seed(433)

        # --- Data Loaders ---
        g = torch.Generator().manual_seed(433)
        train_loader = DataLoader(train_tensor, batch_size=params["batch_size"], shuffle=True, generator=g)
        val_loader = DataLoader(val_tensor, batch_size=params["batch_size"], shuffle=False)

        # --- Model / Optimizer / Scheduler ---
        model = ObjectClassifier(hidden_size=params["hidden_size"], output_size=50, dropout=params["dropout"]).to(device)
        model.initialize_weights(params["init_type"])

        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        # --- TESTING SCHEDULER WARMUP --- #
        if params["warmup_epochs"] > 0:
            # Warmup: linearly increase LR from 1e-8 → base_lr
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=params["warmup_epochs"]
            )

            # Cosine decay for the remaining epochs
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params["num_epochs"] - params["warmup_epochs"]
            )

            # Combine them
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[params["warmup_epochs"]]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

        # --- Training Loop ---
        for epoch in tqdm(range(params["num_epochs"])):
            # --- THIS SCHEDULER STEP PLACEMENT IS VERY IMPORTANT --- #
            if epoch > 0:
                scheduler.step()

            model.train()
            running_loss, correct, total = 0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # --- GAUSSIAN NOISE --- #
                if gauss:
                    noise_std = params['noise_std']
                    X_batch = X_batch + noise_std * torch.randn_like(X_batch)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # --- ATTEMPTING TO FIX INSTABILITY --- #
                # I want to stop the optimizer from taking an update that is too large to recover from.
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            train_acc = 100 * correct / total
            train_accs.append(train_acc)

            losses.append(running_loss / total)

            # --- Validation ---
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    _, predicted = torch.max(outputs, 1)
                    val_total += y_val.size(0)
                    val_correct += (predicted == y_val).sum().item()

            val_acc = 100 * val_correct / val_total
            val_accs.append(val_acc)

            if log:
                print(f"Epoch [{epoch + 1}/{params['num_epochs']}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {running_loss / total}")

            # --- Early Stopping Check ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
                break

            if epoch == 19 and val_acc < 65:
                print(f"Early stopping at epoch {epoch + 1} (insufficient val acc).")
                break

        # --- Load best model weights before returning ---
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

        return model, train_accs, val_accs, losses, best_val_acc

    def train(
            self,
            train_tensor,
            device,
            params,
            grad_clip=True,
            gauss=True,
    ):
        """
        Train the model on the full training dataset.

        Args:
            train_tensor: training data or DataLoader containing features and labels
            device: device to run training on ("cpu" or "cuda")
            params: dictionary of training hyperparameters
            grad_clip: whether to apply gradient clipping during training
            gauss: whether to apply Gaussian noise

        Returns:
            model: the trained model
        """
        self.set_seed(433)

        # --- Data Loaders ---
        g = torch.Generator().manual_seed(433)
        train_loader = DataLoader(train_tensor, batch_size=params["batch_size"], shuffle=True, generator=g)

        # --- Model / Optimizer / Scheduler ---
        model = ObjectClassifier(hidden_size=params["hidden_size"], output_size=50, dropout=params["dropout"]).to(device)
        model.initialize_weights(params["init_type"])

        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        # --- TESTING SCHEDULER WARMUP --- #
        if params["warmup_epochs"] > 0:
            # Warmup: linearly increase LR from 1e-8 → base_lr
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=params["warmup_epochs"]
            )

            # Cosine decay for the remaining epochs
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params["num_epochs"] - params["warmup_epochs"]
            )

            # Combine them
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[params["warmup_epochs"]]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

        # --- Training Loop ---
        for epoch in tqdm(range(params["num_epochs"])):
            # --- THIS SCHEDULER STEP PLACEMENT IS VERY IMPORTANT --- #
            if epoch > 0:
                scheduler.step()

            model.train()

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # --- GAUSSIAN NOISE --- #
                if gauss:
                    noise_std = params['noise_std']
                    X_batch = X_batch + noise_std * torch.randn_like(X_batch)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # --- ATTEMPTING TO FIX INSTABILITY --- #
                # I want to stop the optimizer from taking an update that is too large to recover from.
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

        return model
