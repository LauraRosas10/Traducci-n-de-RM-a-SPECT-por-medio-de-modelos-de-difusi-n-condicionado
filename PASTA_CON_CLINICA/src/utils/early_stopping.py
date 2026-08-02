import numpy as np


class EarlyStopping:
    """
    Early stopping basado en SSIM (mayor es mejor).
    El guardado del modelo lo maneja el Trainer, no esta clase.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_ssim = -np.inf
        self.best_mae = np.inf
        self.counter = 0
        self.best_step = 0
        self.stop_training = False

    def __call__(self, ssim: float, mae: float, step: int) -> bool:
        improved = float(np.mean(ssim)) > self.best_ssim + self.min_delta

        if improved:
            if self.verbose:
                print(f"[EarlyStopping] Step {step}: SSIM {self.best_ssim:.6f} - {ssim:.6f} | MAE={mae:.6f}. Guardando checkpoint...")
            self.best_ssim = ssim
            self.best_mae = mae
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Step {step}: SSIM={ssim:.6f} (mejor={self.best_ssim:.6f}) | MAE={mae:.6f}. Sin mejora {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"[EarlyStopping] Paciencia agotada. Mejor step: {self.best_step} (SSIM={self.best_ssim:.6f}, MAE={self.best_mae:.6f}).")
                self.stop_training = True

        return self.stop_training