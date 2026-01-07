"""
README
-------------------------------------------------------------------------------------------------------------------------------------------------
                             ...-=+**##%%%#*+-:.                                     
                      ...::-....             ..:=*%*-..                              
                   ......   ..:-+*#%%%%%%#+-:..    .:%@+.                            
                  .  ..-+%@@@%#**=-::...:-=++%@@#=..  .*@%+.    .-.                  
                  .=#@@%+-:..   .:-=*####*+-....:=@%=. -@@@=     .*=.                
              ..-@@@+.   ..:=%@@@@@%%%%%%@@@@@@#:. :#%. .=@@.     .%-                
             .*@@*:   ..+%@@%+-..  .=*###+:.=#@@@%: .+#. .#@*.    .%*                
           .+@@=     :@@@*.      .=@@@@@@@@%..*@@@@. .*. .#@*.    :@*                
          .*@#:    :#@@=.    .-*@@@@@%%@@@@@%:%@@@+  .:  =@@:    .%@-                
          =#*.    -@@+.     -@@@@@@@@@#*##@@@@@@@=     .+@@-    .#@*.                
         -@*.    :%@*..:: .=@@@@-=@@@@@@%@@@@%#.     .-%@%:   .:%@*                  
         *@-.   .-@@- .== .=@@@@:.-@@@@@@@@*..     -#@@#:.  .:*@%:                   
         +%:     :%@=..-%:..=@@@@%-..+%%#:....::#@@%#:.  ..-@@@-                     
         .%-     .*@@%=.-@#-..-+@@@@@@@@@@@@@@@%+=..  .:+%@@+:.                      
          :+.    -@@@@@-  :%@@=.   .........     ..+%@@@%:.                          
           .-.   .+@@@@=.   .:+#%@%@%#****##%@@@@%#*=:.   ..                         
                   ....+@*-.    .....::::::::.....    ..:...                         
                         .-*##=-:.....  ........:..-:..                              
                             ...::----------:.....                                   
                         ___  ___  _______   __  _______
--------------------    / _ \/ _ \/ __/ _ | /  |/  / __/   --------------------
====================   / // / , _/ _// __ |/ /|_/ /\ \     ====================
--------------------  /____/_/|_/___/_/ |_/_/  /_/___/     --------------------
"""

# Author: Jack Barry
#       
#       This script contains all the code to train a standard MLP model on the DREAMS dataset in order to predict a SINGLE profile.
#       The path definitions for Y1 and Y2 are the same as those defined in Cov_main.py, however a flag ath the top of "main" allows
#       the user to define if they want to train on Y1 or Y2. Some file paths may need to change. To find those likely to need 
#       changing, Ctrl+f "gjc7gx".The main section of this script contains four sections, "Load Dataset", "Define Hyperparameters", "Build and
#       Train Models", and "Plot". To load a pre trained model, set train = False (at the top of main), and change the model name and
#       structure that of the model you want to load in "Define Hyperparameters" (model structure should be included in the model name in 
#       the form NAME_{cov_hc}_{cov_nl}_{mean_hc}_{mean_nl}.pt. This is not done automatically, so be sure to update the model name 
#       accordingly).


"""
Imports
-------------------------------------------------------------------------------------------------------------------------------------------------"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import h5py, os, optuna, torch
from scipy.spatial import KDTree
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch_geometric as torchg
import torch_scatter as torchs
import torch.multiprocessing as mp
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
import networkx as nx
import pickle
from typing import List, Dict, Tuple, Any
import math
from tqdm import tqdm

"""
Functions
-------------------------------------------------------------------------------------------------------------------------------------------------"""


def norm_params(
    X: np.ndarray,
    groups: List[Dict[str, Any]],   # list of dicts: { "cols": [...], "mode": "linear"|"log" }
    margin_pct: float = 0.0,
    eps: float = 1e-12
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize columns of X in groups, with shared lo/hi/shift per group.
    
    Returns:
        Xn  : normalized array
        key : metadata for denormalization
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    i, j = X.shape
    Xn = np.empty_like(X)
    key = {
        "margin_pct": margin_pct,
        "groups": []
    }

    for group in groups:
        col_indices = group["cols"]
        mode = group.get("mode", "linear")
        subX = X[:, col_indices]

        if mode == "log":
            min_val = np.nanmin(subX)
            shift = max(0.0, -min_val + eps)
            work = np.log(subX + shift + eps)
        else:
            shift = 0.0
            work = subX

        raw_min = np.nanmin(work)
        raw_max = np.nanmax(work)

        if not np.isfinite(raw_min) or not np.isfinite(raw_max):
            raise ValueError(f"Non-finite values encountered in group {col_indices}")

        if np.isclose(raw_max, raw_min):
            lo = raw_min
            hi = raw_min + 1.0
            normed = np.zeros_like(work)
        else:
            span = raw_max - raw_min
            lo = raw_min - margin_pct * span
            hi = raw_max + margin_pct * span
            normed = (work - lo) / (hi - lo)

        Xn[:, col_indices] = normed

        key["groups"].append({
            "cols": col_indices,
            "mode": mode,
            "shift": float(shift),
            "lo": float(lo),
            "hi": float(hi)
        })

    return Xn, key


def denorm_params(Xn: np.ndarray, key: Dict[str, Any]) -> np.ndarray:
    """
    Invert normalization using the group-based key.
    
    Args:
        Xn  : normalized data
        key : dict from norm_params
    Returns:
        X : denormalized data
    """
    Xn = np.asarray(Xn, dtype=float)
    i, j = Xn.shape
    X = np.empty_like(Xn)

    for group in key["groups"]:
        col_indices = group["cols"]
        mode = group["mode"]
        lo = group["lo"]
        hi = group["hi"]
        shift = group["shift"]

        work = Xn[:, col_indices] * (hi - lo) + lo

        if mode == "log":
            X[:, col_indices] = np.exp(work) - shift
        else:
            X[:, col_indices] = work

    return X


class ParamDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def save_checkpoint(path, mean_net, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "mean_state": mean_net.state_dict(),
        "meta":       meta,   # normalization keys, basis info, dims, etc.
    }
    torch.save(payload, path)


def load_checkpoint(path, mean_net, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ckpt] Not found: {path}")
    ckpt = torch.load(path, map_location=device)
    mean_net.load_state_dict(ckpt["mean_state"])
    mean_net.to(device).eval()
    print(f"[ckpt] Loaded model from: {path}")
    return ckpt.get("meta", {})


# ----------------------- Triangular basis helper (unchanged) -----------------------
def triangular_basis_logspace(log_x: np.ndarray,
                            log_knots: np.ndarray,
                            add_bias: bool = True) -> np.ndarray:
    K = len(log_knots)
    Phi = np.zeros((len(log_x), K), dtype=np.float64)
    for j in range(K):
        tj = log_knots[j]
        if j > 0:
            left = log_knots[j - 1]
            Phi[:, j] += np.clip((log_x - left) / (tj - left), 0.0, 1.0)
        if j < K - 1:
            right = log_knots[j + 1]
            Phi[:, j] += np.clip((right - log_x) / (right - tj), 0.0, 1.0)
    if add_bias:
        Phi = np.hstack([np.ones((len(log_x), 1)), Phi])
    return Phi  # [M, K(+1)]

# ----------------------- helper to solve coeffs with optional log-space -----------------------
def solve_coeffs(Y_bins: np.ndarray, Phi: np.ndarray, logspace: bool, eps: float = 1e-12) -> np.ndarray:
    """
    Solve for coefficients c such that:
    - linear   :   Y ≈ Phi @ c
    - logspace : log(Y + eps) ≈ Phi @ c    =>  Y ≈ exp(Phi @ c)
    Vectorized over samples: Y_bins shape [N, M], Phi shape [M, K], returns C shape [N, K].
    """
    if logspace:
        target = np.log(np.clip(Y_bins, eps, None))
    else:
        target = Y_bins
    # Phi: [M,K] ; target^T: [M,N] ; coef^T: [K,N]
    C_T, *_ = np.linalg.lstsq(Phi, target.T, rcond=None)
    return C_T.T  # [N, K]


# ---- Model ----
class MeanMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B, dx]
        return self.net(x)  # [B, dy]


# ---- Train / Eval (mean-only, MSE) ----
mse_loss = nn.MSELoss()

@torch.no_grad()
def eval_epoch(mean_net, loader):
    mean_net.eval()
    tot_sum = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # mask out any rows with NaNs / infs in y
        mask = torch.isfinite(yb).all(dim=1)
        if mask.sum() == 0:
            continue
        xb, yb = xb[mask], yb[mask]

        mu = mean_net(xb)
        loss = mse_loss(mu, yb)
        B = xb.size(0)
        tot_sum += loss.item() * B
        count += B

    return tot_sum / max(1, count)


def train_one_epoch(mean_net, loader, opt, max_grad_norm=1.0):
    mean_net.train()
    tot_sum = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        mask = torch.isfinite(yb).all(dim=1)
        if mask.sum() == 0:
            continue
        xb, yb = xb[mask], yb[mask]

        mu = mean_net(xb)
        loss = mse_loss(mu, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mean_net.parameters(), max_norm=max_grad_norm)
        opt.step()

        B = xb.size(0)
        tot_sum += loss.item() * B
        count += B

    return tot_sum / max(1, count)
    

def reconstruct_profile_from_output(y_vec: np.ndarray,
                                    Yn_key: dict,
                                    basis_meta: dict,
                                    target_block: str) -> np.ndarray:
    """
    Take a single normalized output vector y_vec (length = Y_len),
    denormalize it, and turn it into a 50-bin radial profile for
    the chosen block (Y1 or Y2).
    """
    # 1) denormalize from [0,1] back to original scale of whatever we trained on
    y_vec = np.asarray(y_vec, dtype=float).reshape(1, -1)  # [1, Y_len]
    y_denorm = denorm_params(y_vec, Yn_key)[0]             # [Y_len]

    # 2) If using triangular basis, y_denorm are coefficients -> reconstruct bins
    use_basis = basis_meta.get("use_basis", False)
    if use_basis:
        block_key = "y1" if target_block == "Y1" else "y2"
        block_meta = basis_meta[block_key]
        Phi = block_meta["Phi"]        # [50, K]
        logspace = block_meta["logspace"]

        # y_denorm length should be K (coeffs)
        coeffs = y_denorm[: Phi.shape[1]]
        profile = Phi @ coeffs         # [50]
        if logspace:
            profile = np.exp(profile)
        return profile

    # 3) Raw-bins mode: y_denorm already is the 50-bin profile
    return y_denorm

# ----------------------- Plot helper -----------------------
def plot_one_test_example(mean_net,
                            test_loader,
                            Yn_key,
                            basis_meta,
                            r_tags,
                            target_block: str,
                            use_tri_basis: bool,
                            device,
                            yscale: str = "linear"):
    """
    Plot the first sample in the test_loader:
    - True profile vs predicted profile
    - y-axis can be 'linear' or 'log'
    """
    mean_net.eval()

    # Grab first batch and first element
    xb, yb = next(iter(test_loader))   # xb: [B, X_len], yb: [B, Y_len] (normalized)
    xb = xb.to(device)
    yb = yb.to(device)

    with torch.no_grad():
        y_pred = mean_net(xb)          # [B, Y_len]

    # Move to CPU / numpy
    y_true_0 = yb[0].cpu().numpy()
    y_pred_0 = y_pred[0].cpu().numpy()

    # Reconstruct physical profiles (50 bins on r_tags)
    true_profile = reconstruct_profile_from_output(
        y_true_0, Yn_key, basis_meta, target_block
    )
    pred_profile = reconstruct_profile_from_output(
        y_pred_0, Yn_key, basis_meta, target_block
    )

    # Basic labels
    if target_block == "Y1":
        y_label = r"$\rho(r)$ (Y1)"
    else:
        y_label = r"$v_c(r)$ (Y2)"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_tags, true_profile, label="True", linewidth=2.0)
    ax.plot(r_tags, pred_profile, label="Predicted", linestyle="--", linewidth=2.0)

    ax.set_xscale("log")  # radius is log-spaced
    ax.set_xlabel(r"$r\ \mathrm{[kpc]}$")
    ax.set_ylabel(y_label)
    ax.set_title(f"Test example: {target_block} profile (yscale = {yscale})")

    if yscale.lower() == "log":
        ax.set_yscale("log")

    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle=":")
    plt.tight_layout()

    # Save figure
    fig_dir = "/home/gjc7gx/CovNNet/Figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(
        fig_dir, f"one_test_example_{target_block}_{yscale}.png"
    )
    plt.savefig(fig_path, dpi=160)
    print(f"[plot] Saved test example figure to: {fig_path}")

# ----------------------- Plot ALL test profiles + metric -----------------------
def plot_all_test_profiles_and_error(mean_net,
                                        test_loader,
                                        Yn_key,
                                        basis_meta,
                                        r_tags,
                                        target_block: str,
                                        device,
                                        yscale: str = "linear"):
    """
    1) For every sample in the test_loader:
        - Reconstruct true & predicted physical profiles (50 bins)
        - Plot all true and predicted curves on a single figure
    2) Compute the scalar metric:
            mean_{halos,r} | log( pred(r) / true(r) ) |
        over all test samples and radii where both pred & true > 0.
    """
    mean_net.eval()

    fig, ax = plt.subplots(figsize=(7, 5))

    total_abs_log = 0.0
    total_count = 0

    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.no_grad():
            y_pred = mean_net(xb)  # [B, Y_len]

        y_true_np = yb.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        B = y_true_np.shape[0]
        for i in range(B):
            # Reconstruct 50-bin profiles
            true_profile = reconstruct_profile_from_output(
                y_true_np[i], Yn_key, basis_meta, target_block
            )
            pred_profile = reconstruct_profile_from_output(
                y_pred_np[i], Yn_key, basis_meta, target_block
            )

            # Plot with low alpha so many curves can be overlaid
            ax.plot(r_tags, pred_profile/true_profile, alpha=0.25, linewidth=1.0)

            # Accumulate |log(pred/true)|, but only where both > 0
            true_arr = np.asarray(true_profile, dtype=float)
            pred_arr = np.asarray(pred_profile, dtype=float)
            mask = (true_arr > 0.0) & (pred_arr > 0.0)

            if np.any(mask):
                eps = 1e-8
                ratio = (pred_arr[mask] + eps) / (true_arr[mask] + eps)
                abs_log = np.abs(np.log(ratio))
                total_abs_log += abs_log.sum()
                total_count += abs_log.size

    # Configure axes
    ax.set_xscale("log")
    if yscale.lower() == "log":
        ax.set_yscale("log")

    if target_block == "Y1":
        y_label = r"$\rho(r)$ (Y1)"
    else:
        y_label = r"$v_c(r)$ (Y2)"

    ax.set_xlabel(r"$r\ \mathrm{[kpc]}$")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"All test-set {target_block} profiles\n"
        f"(solid=true, dashed=predicted, yscale={yscale})"
    )
    ax.grid(True, which="both", linestyle=":")

    plt.tight_layout()

    fig_dir = "/home/gjc7gx/CovNNet/Figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(
        fig_dir, f"all_test_profiles_{target_block}_{yscale}.png"
    )
    plt.savefig(fig_path, dpi=160)
    print(f"[plot] Saved all-test-profiles figure to: {fig_path}")

    # Final scalar metric
    if total_count > 0:
        mean_abs_log = total_abs_log / float(total_count)
    else:
        mean_abs_log = float("nan")

    print(f"[metric] mean |log(pred/true)| over test set = {mean_abs_log:.6g}")
    return mean_abs_log


def save_test_profiles_txt(mean_net,
                        test_loader,
                        Yn_key,
                        basis_meta,
                        r_tags,
                        target_block: str,
                        device,
                        out_path: str):
    """
    Saves TRUE and PREDICTED physical profiles for the entire test set.

    Output file columns:
        [ sample_index , 50 TRUE profile bins , 50 PRED profile bins ]
    """

    mean_net.eval()
    all_rows = []
    sample_counter = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            y_pred = mean_net(xb)

            y_true_np = yb.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()

            B = y_true_np.shape[0]
            for i in range(B):
                true_profile = reconstruct_profile_from_output(
                    y_true_np[i], Yn_key, basis_meta, target_block
                )
                pred_profile = reconstruct_profile_from_output(
                    y_pred_np[i], Yn_key, basis_meta, target_block
                )

                row = np.concatenate((
                    np.array([sample_counter], dtype=float),
                    true_profile.astype(float),
                    pred_profile.astype(float)
                ))

                all_rows.append(row)
                sample_counter += 1

    all_rows = np.vstack(all_rows)

    header = (
        "sample_index  "
        + "  ".join([f"true_r{i}" for i in range(len(r_tags))])
        + "  "
        + "  ".join([f"pred_r{i}" for i in range(len(r_tags))])
    )

    np.savetxt(out_path, all_rows, header=header)
    print(f"[save] Saved test profiles to:\n{out_path}")

def init_print():
    print("""
                             ...-=+**##%%%#*+-:.                                     
                      ...::-....             ..:=*%*-..                              
                   ......   ..:-+*#%%%%%%#+-:..    .:%@+.                            
                  .  ..-+%@@@%#**=-::...:-=++%@@#=..  .*@%+.    .-.                  
                  .=#@@%+-:..   .:-=*####*+-....:=@%=. -@@@=     .*=.                
              ..-@@@+.   ..:=%@@@@@%%%%%%@@@@@@#:. :#%. .=@@.     .%-                
             .*@@*:   ..+%@@%+-..  .=*###+:.=#@@@%: .+#. .#@*.    .%*                
           .+@@=     :@@@*.      .=@@@@@@@@%..*@@@@. .*. .#@*.    :@*                
          .*@#:    :#@@=.    .-*@@@@@%%@@@@@%:%@@@+  .:  =@@:    .%@-                
          =#*.    -@@+.     -@@@@@@@@@#*##@@@@@@@=     .+@@-    .#@*.                
         -@*.    :%@*..:: .=@@@@-=@@@@@@%@@@@%#.     .-%@%:   .:%@*                  
         *@-.   .-@@- .== .=@@@@:.-@@@@@@@@*..     -#@@#:.  .:*@%:                   
         +%:     :%@=..-%:..=@@@@%-..+%%#:....::#@@%#:.  ..-@@@-                     
         .%-     .*@@%=.-@#-..-+@@@@@@@@@@@@@@@%+=..  .:+%@@+:.                      
          :+.    -@@@@@-  :%@@=.   .........     ..+%@@@%:.                          
           .-.   .+@@@@=.   .:+#%@%@%#****##%@@@@%#*=:.   ..                         
                   ....+@*-.    .....::::::::.....    ..:...                         
                         .-*##=-:.....  ........:..-:..                              
                             ...::----------:.....                                   
                         ___  ___  _______   __  _______
--------------------    / _ \/ _ \/ __/ _ | /  |/  / __/   --------------------
====================   / // / , _/ _// __ |/ /|_/ /\ \     ====================
--------------------  /____/_/|_/___/_/ |_/_/  /_/___/     --------------------
""")
    return


"""
Main
-------------------------------------------------------------------------------------------------------------------------------------------------"""

if __name__ == "__main__":

    init_print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = True
    plot = True

    # ----------------------- Target block -----------------------
    # Choose which block the network should learn: "Y1" (density) or "Y2" (rotation)
    TARGET_BLOCK = "Y2"  # change to "Y2" when you want to train on Y2 only

    """
    Load Dataset
    -------------------------------------------------------------------------------------------------------------------------------------------------"""

    # ----------------------- Config -----------------------
    USE_TRI_BASIS = True       # <--- toggle: False = raw 50+50 bins, True = basis coeffs
    K1_COEFFS     = 16         # number of coeffs for Y1 (density)
    K2_COEFFS     = 16         # number of coeffs for Y2 (rotation)

    # ----------------------- Seed -------------------------
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------- Inputs -----------------------
    params_path = "/standard/DREAMS/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt"
    mass_path   = "/home/gjc7gx/CovNNet/Params/mass_params.txt"

    params = np.loadtxt(params_path)[:, :]
    masses = np.loadtxt(mass_path)[:, 1:]
    X = np.column_stack((params, masses))

    """{"cols": list(range(0, 1)), "mode": "linear"},
    {"cols": list(range(1, 2)), "mode": "linear"},
    {"cols": list(range(2, 3)), "mode": "linear"},
    {"cols": list(range(3, 4)), "mode": "linear"},"""

    X_groups = [
        {"cols": list(range(0, 5)), "mode": "linear"},
        {"cols": list(range(5, 6)), "mode": "log"},
        {"cols": list(range(6, 7)), "mode": "linear"}
    ]
    Xn, Xn_key = norm_params(X, X_groups, margin_pct=0.05)

    # ----------------------- Outputs (raw bins) -----------------------
    y1_path = "/home/gjc7gx/CovNNet/Params/denProf_pts_params.txt"
    Y1 = np.loadtxt(y1_path)[:, 1:]  # [N, 50]

    y2_path = "/home/gjc7gx/CovNNet/Params/rotCurve_pts_params.txt"
    Y2 = np.loadtxt(y2_path)[:, 1:]  # [N, 50]

    # common radius grid for both blocks (50 bins)
    r_tags = np.logspace(np.log10(0.4), np.log10(50), 50)

    assert len(X) == len(Y1) == len(Y2), "Mismatch in number of samples."

     # ----------------------- Switch: raw vs basis -----------------------
    basis_meta = {"use_basis": USE_TRI_BASIS, "target_block": TARGET_BLOCK}

    if USE_TRI_BASIS:
        # Choose whether each block uses log-space basis fitting
        LOG_Y1_BASIS = True    # density is strictly positive -> log is natural
        LOG_Y2_BASIS = False   # set True if you also want log for rotation curve

        # Build triangular bases (log-spaced knots) on the 50-bin grid
        log_r = np.log(r_tags)
        knots1 = np.linspace(log_r.min(), log_r.max(), K1_COEFFS)   # density
        knots2 = np.linspace(log_r.min(), log_r.max(), K2_COEFFS)   # rotation

        Phi1 = triangular_basis_logspace(log_r, knots1, add_bias=True)  # [50, K1+1]
        Phi2 = triangular_basis_logspace(log_r, knots2, add_bias=True)  # [50, K2+1]

        # Solve coefficients (optionally in log-space)
        C1 = solve_coeffs(Y1, Phi1, logspace=LOG_Y1_BASIS)  # [N, K1+1]
        C2 = solve_coeffs(Y2, Phi2, logspace=LOG_Y2_BASIS)  # [N, K2+1]

        if TARGET_BLOCK == "Y1":
            # Learn only the Y1 coefficients
            Y = C1
            Y_groups = [
                {"cols": list(range(0, C1.shape[1])), "mode": "linear"},
            ]
        elif TARGET_BLOCK == "Y2":
            # Learn only the Y2 coefficients
            Y = C2
            Y_groups = [
                {"cols": list(range(0, C2.shape[1])), "mode": "linear"},
            ]
        else:
            raise ValueError(f"Unknown TARGET_BLOCK={TARGET_BLOCK}, must be 'Y1' or 'Y2'.")

        Y_len = Y.shape[1]
        Yn, Yn_key = norm_params(Y, Y_groups, margin_pct=0.02)

        # Save meta so we can reconstruct correctly (linear vs log) later
        basis_meta.update({
            "y1": {"Phi": Phi1, "k": C1.shape[1], "r_tags": r_tags, "logspace": LOG_Y1_BASIS},
            "y2": {"Phi": Phi2, "k": C2.shape[1], "r_tags": r_tags, "logspace": LOG_Y2_BASIS},
        })

    else:
        # Raw 50-bin blocks
        if TARGET_BLOCK == "Y1":
            Y = Y1  # [N, 50]
            Y_groups = [
                {"cols": list(range(0, 50)), "mode": "log"},  # density -> log
            ]
        elif TARGET_BLOCK == "Y2":
            Y = Y2  # [N, 50]
            Y_groups = [
                {"cols": list(range(0, 50)), "mode": "linear"},  # rotation -> linear
            ]
        else:
            raise ValueError(f"Unknown TARGET_BLOCK={TARGET_BLOCK}, must be 'Y1' or 'Y2'.")

        Yn, Yn_key = norm_params(Y, Y_groups, margin_pct=0.02)
        Y_len = Y.shape[1]
        basis_meta.update({"y1": None, "y2": None})


    # --------------- final sanity ---------------
    assert Xn.shape[0] == Yn.shape[0], "Mismatch in #samples after preprocessing"


    # Train-validation-test split
    train_size = 0.8
    valid_size = 0.1
    test_size  = 0.1
    batch_size = 32

    print("Creating dataset with seed " + str(seed) + "...")
    dataset = ParamDataset(Xn, Yn)
    n_total = len(dataset)
    n_train = int(train_size * n_total)
    n_valid = int(valid_size * n_total)
    n_test  = n_total - n_train - n_valid

    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_data, valid_data, test_data = random_split(dataset, [n_train, n_valid, n_test], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader  = DataLoader(test_data, batch_size=batch_size)
    print("Done")

    # Step 1: Save indices
    train_indices = train_data.indices if hasattr(train_data, 'indices') else [i for i in range(len(train_data))]
    valid_indices = valid_data.indices if hasattr(valid_data, 'indices') else [i for i in range(len(valid_data))]
    test_indices  = test_data.indices  if hasattr(test_data,  'indices')  else [i for i in range(len(test_data))]

    # Store them for later check (e.g., write to file or print hash)
    print()
    print("Checking Dataset")
    hashVals42 = [-7122228232448510862, -6337307882137213621,-6788134668364437632]
    hashVals =[hash(tuple(train_indices)), hash(tuple(valid_indices)), hash(tuple(test_indices))]

    if seed == 42 and hashVals == hashVals42:
        print("COMPLETE")
        print()
    else:
        print("WARNING: Dataset may be inconsistant. There may be crossover between training and test data.")
        print()

    """
    Define Hyperparamiters
    -------------------------------------------------------------------------------------------------------------------------------------------------"""

    MODEL_DIR  = "/home/gjc7gx/CovNNet/Models"
    MODEL_NAME = f"Test"
    CKPT_PATH  = os.path.join(MODEL_DIR, MODEL_NAME + ".pt")

    # -------------------------
    # Hyperparameters & build
    # -------------------------
    mean_hc       = 200   # hidden width
    mean_nl       = 4     # hidden layers
    wd       = 1e-5
    n_epochs = 35000
    
    # Def model input and output dim
    X_len = X.shape[1]
    Y1_len = Y1.shape[1]
    Y2_len = Y2.shape[1]

    """
    Build and Train Models
    -------------------------------------------------------------------------------------------------------------------------------------------------"""

    # Build model and optimizer (mean-only)
    mean_net = MeanMLP(in_dim=X.shape[1], out_dim=Y_len, hidden=mean_hc, n_layers=mean_nl).to(device)

    optimizer = optim.Adam(mean_net.parameters(), lr=1e-4, weight_decay=wd)

    print(f"Training (mean-only) with output dim = {Y_len}")
    best_val = float("inf")
    best_state = None

    if train:

        train_losses, val_losses = [], []
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Training vs Validation Loss (mean only)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale("log")
        (line_train,) = ax.plot([], [], label="Train", lw=1.8)
        (line_val,)   = ax.plot([], [], label="Validation", lw=1.8)
        ax.legend(loc="best")
        plt.ion(); plt.show(block=False)

        for epoch in range(1, n_epochs + 1):
            tr_loss = train_one_epoch(mean_net, train_loader, optimizer)
            va_loss = eval_epoch(mean_net, valid_loader)

            if epoch % 10 == 0 or epoch == 1:
                if va_loss < best_val:
                    print(f"Epoch {epoch:04d} | Train: {tr_loss:.6f} | Val: {va_loss:.6f} (B)")
                else:
                    print(f"Epoch {epoch:04d} | Train: {tr_loss:.6f} | Val: {va_loss:.6f}")

            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            # Update plot every N epochs (e.g. every 100)
            if epoch % 100 == 0:
                line_train.set_data(range(1, len(train_losses)+1), train_losses)
                line_val.set_data(range(1, len(val_losses)+1), val_losses)
                ax.relim(); ax.autoscale_view()
                plt.draw(); plt.pause(0.001)
                plt.savefig("/home/gjc7gx/CovNNet/Figures/ALLloss_curve_mean_only.png", dpi=160)

            if va_loss < best_val:
                best_val = va_loss
                best_state = mean_net.state_dict()

                mean_net.load_state_dict(best_state)

                meta = {
                    "Xn_key": Xn_key,                # normalization key for inputs
                    "Yn_key": Yn_key,                # normalization key for outputs (coeffs or bins)
                    "basis_meta": basis_meta,        # contains Phi1/Phi2, logspace flags, etc.
                    "r_tags": r_tags.tolist(),       # for plotting
                    "dims": {"X": X_len, "Y": Y_len},
                    "target_block": TARGET_BLOCK,
                }
                save_checkpoint(CKPT_PATH, mean_net, meta)

    else:
        # No training: just load the existing mean-only checkpoint
        loaded_meta = load_checkpoint(CKPT_PATH, mean_net, device)
        # Prefer keys/metadata from the checkpoint if present
        Yn_key     = loaded_meta.get("Yn_key", Yn_key)
        Xn_key     = loaded_meta.get("Xn_key", Xn_key)
        basis_meta = loaded_meta.get("basis_meta", basis_meta)
        if "r_tags" in loaded_meta:
            r_tags = np.array(loaded_meta["r_tags"], dtype=float)
        TARGET_BLOCK = loaded_meta.get("target_block", TARGET_BLOCK)

    """
    Plot
    -------------------------------------------------------------------------------------------------------------------------------------------------"""
    if plot:
        plot_one_test_example(
            mean_net=mean_net,
            test_loader=test_loader,
            Yn_key=Yn_key,
            basis_meta=basis_meta,
            r_tags=r_tags,
            target_block=TARGET_BLOCK,
            use_tri_basis=USE_TRI_BASIS,
            device=device,
            yscale="log",
        )

        # All test profiles + mean |log(pred/true)|
        mean_abs_log = plot_all_test_profiles_and_error(
            mean_net=mean_net,
            test_loader=test_loader,
            Yn_key=Yn_key,
            basis_meta=basis_meta,
            r_tags=r_tags,
            target_block=TARGET_BLOCK,
            device=device,
            yscale="log",
        )
        print(f"[metric] Final mean |log(pred/true)| = {mean_abs_log:.6g}")

        out_txt = "/home/gjc7gx/CovNNet/Params/Y2O_ALL_PredTrue.txt"
        save_test_profiles_txt(
            mean_net=mean_net,
            test_loader=test_loader,
            Yn_key=Yn_key,
            basis_meta=basis_meta,
            r_tags=r_tags,
            target_block=TARGET_BLOCK,
            device=device,
            out_path=out_txt
)


        

