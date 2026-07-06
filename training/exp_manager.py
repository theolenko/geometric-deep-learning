import os
import json
import torch


class ExpManager:
    def __init__(self, exp_dir, run_cfg, patience=None, checkpoint_every=10):
        self.exp_dir = exp_dir
        self.run_cfg = run_cfg
        self.patience = patience                 # None = no early stopping
        self.checkpoint_every = checkpoint_every

        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.epochs_without_improvement = 0

        self.train_losses_l1 = []
        self.train_losses_mae = []
        self.val_losses = []

        # 1) create the experiment dir + 2) save the run config
        os.makedirs(self.exp_dir, exist_ok=True)
        self._save_config()
        print(f"[ExpManager] Saving to: {os.path.abspath(self.exp_dir)}")

    def _save_config(self):
        path = os.path.join(self.exp_dir, "run_config.json")
        with open(path, "w") as f:
            # default=str lets non-JSON objects (e.g. torch dtypes) serialize gracefully
            json.dump(self.run_cfg, f, indent=2, default=str)

    def save_losses(self):
        path = os.path.join(self.exp_dir, "losses.json")
        with open(path, "w") as f:
            json.dump({
                "train_losses_l1": self.train_losses_l1,
                "train_losses_mae": self.train_losses_mae,
                "val_losses": self.val_losses,
            }, f, indent=2)

    def _save(self, path, model, epoch, optimizer=None, extra=None):
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    # --- function: save the best model ---
    def save_best_model(self, model, epoch, optimizer=None, extra=None):
        path = os.path.join(self.exp_dir, "best_model.pt")
        self._save(path, model, epoch, optimizer, extra)
        self.save_losses()
        print(f"[ExpManager] Saved best model (epoch {epoch}, "
                f"val_loss={self.best_val_loss:.6f}) -> {path}")

    def save_checkpoint(self, model, epoch, optimizer=None, extra=None):
        path = os.path.join(self.exp_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        self._save(path, model, epoch, optimizer, extra)
        self.save_losses()
        print(f"[ExpManager] Saved checkpoint (epoch {epoch}) -> {path}")

    # --- function: check if better than best, and if so save ---
    def check_improvement(self, val_loss, model, epoch, optimizer=None):
        improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.save_best_model(model, epoch, optimizer)
            print(f"[ExpManager] New best at epoch {epoch}: val_loss={val_loss:.6f}")
        else:
            self.epochs_without_improvement += 1
        return improved

    # --- function: check if patience ran out ---
    def should_stop(self):
        if self.patience is None:
            return False
        return self.epochs_without_improvement >= self.patience

    # convenience: one call per epoch that ties it all together
    def step(self, val_loss, model, epoch, optimizer=None, train_loss_l1=None, train_loss_mae=None):
        """Returns True if training should stop (early stopping)."""
        self.val_losses.append(val_loss)
        if train_loss_l1 is not None:
            self.train_losses_l1.append(train_loss_l1)
        if train_loss_mae is not None:
            self.train_losses_mae.append(train_loss_mae)

        self.check_improvement(val_loss, model, epoch, optimizer)
        if self.checkpoint_every and epoch % self.checkpoint_every == 0:
            self.save_checkpoint(model, epoch, optimizer)
        return self.should_stop()