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


def save_checkpoint(path, mean_net, cov_net, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "mean_state": mean_net.state_dict(),
        "cov_state":  cov_net.state_dict(),
        "meta":       meta,   # normalization keys, basis info, dims, etc.
    }
    torch.save(payload, path)
    # print(f"[ckpt] Saved best model to: {path}")

def load_checkpoint(path, mean_net, cov_net, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ckpt] Not found: {path}")
    ckpt = torch.load(path, map_location=device)
    mean_net.load_state_dict(ckpt["mean_state"])
    cov_net.load_state_dict(ckpt["cov_state"])
    mean_net.to(device).eval()
    cov_net.to(device).eval()
    print(f"[ckpt] Loaded model from: {path}")
    return ckpt.get("meta", {})


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
    mass_path   = "/home/gjc7gx/CovNNet/Params/Other_mass_params.txt"

    params = np.loadtxt(params_path)[:, :]
    masses = np.loadtxt(mass_path)[:, 1:]
    X = np.column_stack((params, masses))
    # X = params

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
    y1_path = "/home/gjc7gx/CovNNet/Params/Other_denProf_pts_params.txt"
    Y1 = np.loadtxt(y1_path)[:, 1:]  # [N, 50]

    y2_path = "/home/gjc7gx/CovNNet/Params/rotCurve_pts_params.txt"
    Y2 = np.loadtxt(y2_path)[:, 1:]  # [N, 50]

    # common radius grid for both blocks (50 bins)
    r_tags = np.logspace(np.log10(0.4), np.log10(50), 50)

    assert len(X) == len(Y1) == len(Y2), "Mismatch in number of samples."

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

    basis_meta = {"use_basis": USE_TRI_BASIS}

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

        # New learning targets are the concatenated coefficients
        Y = np.hstack([C1, C2])  # [N, (K1+1)+(K2+1)]
        Y_len = Y.shape[1]

        # Coefficients can be negative; keep normalization linear in coeff space
        Y_groups = [
            {"cols": list(range(0, C1.shape[1])), "mode": "linear"},                          # coeffs for Y1
            {"cols": list(range(C1.shape[1], C1.shape[1] + C2.shape[1])), "mode": "linear"},  # coeffs for Y2
        ]
        Yn, Yn_key = norm_params(Y, Y_groups, margin_pct=0.02)

        # Save meta so we can reconstruct correctly (linear vs log) later
        basis_meta.update({
            "y1": {"Phi": Phi1, "k": C1.shape[1], "r_tags": r_tags, "logspace": LOG_Y1_BASIS},
            "y2": {"Phi": Phi2, "k": C2.shape[1], "r_tags": r_tags, "logspace": LOG_Y2_BASIS},
        })

    else:
        # Raw 50+50 bins as before
        Y = np.column_stack((Y1, Y2))  # [N, 100]
        Y_groups = [
            {"cols": list(range(0, 50)), "mode": "log"},     # density -> log
            {"cols": list(range(50, 100)), "mode": "linear"} # rotation -> linear
        ]
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
    Load Dataset
    -------------------------------------------------------------------------------------------------------------------------------------------------"""


    MODEL_DIR  = "/home/gjc7gx/CovNNet/Models"
    MODEL_NAME = f"Other_3_64_4_200"
    CKPT_PATH  = os.path.join(MODEL_DIR, MODEL_NAME + ".pt")

    # -------------------------
    # Hyperparameters & build
    # -------------------------
    FULL_COV = True  # keep True to learn full correlation structure
    diag_eps = 1e-6
    cov_hc       = 64    # hidden width
    cov_nl       = 3     # hidden layers
    mean_hc       = 200   # hidden width
    mean_nl       = 4     # hidden layers
    wd       = 1e-5
    n_epochs = 35000
    
    # Def model input and output dim
    X_len = X.shape[1]
    Y1_len = Y1.shape[1]
    Y2_len = Y2.shape[1]
    Y_len = Y.shape[1]

    """
    Load Dataset
    -------------------------------------------------------------------------------------------------------------------------------------------------"""


    # ---- utils: triangular handling and (de)normalization for mean/cov ----
    def _tril_indices_cached(d: int, device):
        rows, cols = torch.tril_indices(d, d, 0, device=device)
        diag_mask = rows == cols
        return rows, cols, diag_mask

    def denorm_mean_and_cholesky(mu_n: torch.Tensor,
                                L_n: torch.Tensor,
                                key: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert normalized mean & Cholesky to original scale for plotting.
        Assumes per-dimension affine normalization (mode='linear' in your key).
        For each dim j: y = (hi - lo) * y_n + lo  =>  Σ = D Σ_n D  with D = diag(hi - lo).
        Thus, L = D @ L_n is a valid Cholesky for Σ.
        """
        assert key["mode"] == "linear", "denorm of full Σ only implemented for linear mode."
        device = mu_n.device
        d = mu_n.shape[-1]

        # build diagonal scale and offset vectors
        scales = torch.tensor([c["hi"] - c["lo"] for c in key["cols"]],
                            dtype=mu_n.dtype, device=device)  # [d]
        offsets = torch.tensor([c["lo"] for c in key["cols"]],
                            dtype=mu_n.dtype, device=device)  # [d]

        # mean
        mu = mu_n * scales + offsets
        # cholesky
        D = torch.diag(scales)  # [d,d]
        L = D @ L_n.squeeze(0)  # [d,d]
        L = L.unsqueeze(0)      # [1,d,d]
        return mu, L

    # ---- Models ----
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
            return self.net(x) # [B, dy]


    class CovWithTemp(nn.Module):
        def __init__(self, cov_head):
            super().__init__()
            self.cov_head = cov_head
            self.log_tau = nn.Parameter(torch.zeros(()))  # tau starts at 1.0

        def forward(self, x):
            L = self.cov_head(x)
            tau = torch.exp(self.log_tau)
            return L * tau


    class CovarianceMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=256, n_layers=3,
                    min_std=5e-5, max_std=None, off_cap=0.3, off_warmup_scale=1.0,
                    diag_temp=0.8, **unused):
            super().__init__()
            self.dy = out_dim
            self.min_std = float(min_std)
            self.max_std = None if max_std is None else float(max_std)
            self.off_cap = float(off_cap)
            self.register_buffer("off_warmup_scale", torch.tensor(float(off_warmup_scale)))
            self.diag_temp = float(diag_temp)

            m = out_dim * (out_dim + 1) // 2
            layers, last = [], in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(last, hidden), nn.ReLU()]
                last = hidden
            self.net = nn.Sequential(*layers, nn.Linear(last, m))

            rows, cols = torch.tril_indices(out_dim, out_dim, 0)
            diag_mask = rows == cols
            self.register_buffer("rows", rows)
            self.register_buffer("cols", cols)
            self.register_buffer("diag_mask", diag_mask)
            self.register_buffer("off_mask", ~diag_mask)

            # init around a moderate std (e.g., 0.1)
            target_std = 0.1
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
            with torch.no_grad():
                bias = self.net[-1].bias
                # raw diag centered so softplus(raw/diag_temp) ≈ target_std - min_std
                target_raw = math.log(math.expm1(max(target_std - self.min_std, 1e-6)))
                bias[self.diag_mask] = target_raw * self.diag_temp

        def forward(self, x):
            B, d = x.shape[0], self.dy
            params = self.net(x)
            raw_diag = params[:, self.diag_mask]                 # [B,d]
            raw_off  = params[:, self.off_mask]                  # [B,m-d]

            # Diagonal via softplus: positive and good gradients
            diag = self.min_std + F.softplus(raw_diag / self.diag_temp)
            if self.max_std is not None:
                diag = torch.clamp(diag, max=self.max_std)

            # Off-diagonals (you can keep tanh)
            off  = torch.tanh(raw_off) * self.off_cap * float(self.off_warmup_scale.item())

            L_off = x.new_zeros(B, d, d)
            L_off[:, self.rows[self.off_mask], self.cols[self.off_mask]] = off
            L = L_off + torch.diag_embed(diag)
            return L


    def gaussian_nll(mu, L, y, cov_weight=1.0, lambda_off=1e-3):
        diff = (y - mu).unsqueeze(-1)
        z = torch.linalg.solve_triangular(L, diff, upper=False)
        maha   = (z.square()).sum(dim=(1,2))
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        dconst = L.size(-1) * math.log(2.0 * math.pi)

        mean_term = 0.5 * maha
        cov_term  = 0.5 * (logdet + dconst)

        diag = torch.diagonal(L, dim1=1, dim2=2)
        off  = L - torch.diag_embed(diag)
        off_pen = off.pow(2).mean(dim=(1,2))  # encourage small off-diagonals early

        total = mean_term + cov_weight * cov_term + lambda_off * off_pen
        return total.mean(), mean_term.mean(), cov_term.mean()


    # ---- Train / Eval ----
    def make_models(in_dim, out_dim, cov_hc=256, mean_hc=256, cov_nl=3, mean_nl=3, diag_eps=1e-6):
        mean_net = MeanMLP(in_dim, out_dim, hidden=mean_hc, n_layers=mean_nl).to(device)
        base_cov = CovarianceMLP(in_dim=X_len, out_dim=Y_len, hidden=cov_hc, n_layers=cov_nl, diag_eps=1e-6).to(device)
        cov_net  = CovWithTemp(base_cov).to(device)

        return mean_net, cov_net, gaussian_nll


    @torch.no_grad()
    def eval_epoch(mean_net, cov_net, loss_fn, loader):
        mean_net.eval(); cov_net.eval()
        tot_sum = mean_sum = cov_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu = mean_net(xb)
            L  = cov_net(xb)
            B  = xb.size(0)
            total_loss, mean_loss, cov_loss = loss_fn(mu, L, yb)

            tot_sum  += total_loss.item() * B
            mean_sum += mean_loss.item()  * B
            cov_sum  += cov_loss.item()   * B
            count    += B

        avg_total = tot_sum / max(1, count)
        avg_mean  = mean_sum / max(1, count)
        avg_cov   = cov_sum  / max(1, count)
        return avg_total, avg_mean, avg_cov


    def train_one_epoch(mean_net, cov_net, loss_fn, loader, opt, max_grad_norm=1.0):
        mean_net.train(); cov_net.train()
        tot_sum = mean_sum = cov_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            mask = torch.isfinite(yb).all(dim=1)
            if mask.sum() == 0:
                continue
            xb, yb = xb[mask], yb[mask]

            mu = mean_net(xb)
            L  = cov_net(xb)

            B  = xb.size(0)
            try:
                total_loss, mean_loss, cov_loss = loss_fn(mu, L, yb)
            except FloatingPointError:
                total_loss, mean_loss, cov_loss = loss_fn(mu, L, yb, jitter=1e-8)

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(mean_net.parameters()) + list(cov_net.parameters()),
                                        max_norm=max_grad_norm)
            opt.step()

            tot_sum  += total_loss.item() * B
            mean_sum += mean_loss.item()  * B
            cov_sum  += cov_loss.item()   * B
            count    += B

        avg_total = tot_sum / max(1, count)
        avg_mean  = mean_sum / max(1, count)
        avg_cov   = cov_sum  / max(1, count)
        return avg_total, avg_mean, avg_cov


    @torch.no_grad()
    def epoch_mean_matrix(cov_net, loader, device, which="Sigma"):
        """
        Returns the mean matrix over the loader for this epoch:
        which: "Sigma" (L L^T), "Corr" (normalized Sigma), or "L" (Cholesky).
        """
        cov_net.eval()
        accum = None
        count = 0
        for xb, _ in loader:
            xb = xb.to(device)
            L = cov_net(xb)                           # [B,d,d]
            if which == "L":
                M = L
            else:
                Sigma = L @ L.transpose(1, 2)         # [B,d,d]
                if which == "Sigma":
                    M = Sigma
                elif which == "Corr":
                    std = torch.sqrt(torch.clamp(torch.diagonal(Sigma, dim1=1, dim2=2), 1e-18))
                    denom = std.unsqueeze(2) * std.unsqueeze(1)  # [B,d,1]*[B,1,d]
                    M = Sigma / denom
                else:
                    raise ValueError("which must be 'Sigma', 'Corr', or 'L'")
            batch_mean = M.mean(dim=0)                 # [d,d]
            accum = batch_mean if accum is None else accum + batch_mean
            count += 1
        return (accum / count).cpu().numpy()           # [d,d]


    def plot_matrix_deltas(mean_mats, titles=None):
        """
        mean_mats: list of [d,d] epoch-mean matrices in chronological order
        titles: optional list of titles for the 4 panels
        """
        assert len(mean_mats) >= 2, "Need at least 2 epochs"
        n = len(mean_mats)
        ref = mean_mats[0]
        # choose 4 epochs to show
        idxs = [0, max(1, n//3), max(2, 2*n//3), n-1]
        # compute deltas vs epoch 1
        deltas = [mean_mats[i] - ref for i in idxs]

        # symmetric color scale across all deltas
        vmax = max(np.max(np.abs(D)) for D in deltas)
        vmin = -vmax

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        for ax, D, i in zip(axes, deltas, idxs):
            im = ax.imshow(D, cmap="bwr", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
            ax.set_title(titles[i] if titles else f"Δ vs epoch 1 (epoch {i+1})")
            ax.set_xlabel("col"); ax.set_ylabel("row")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
        cbar.set_label("Change relative to epoch 1")
        plt.savefig("matDeltas.png")

    mean_net, cov_net, loss_fn = make_models(
        in_dim=X.shape[1], out_dim=Y_len, cov_hc=cov_hc, cov_nl=cov_nl, mean_hc=mean_hc, mean_nl=mean_nl,diag_eps=diag_eps
    )
    # Separate learning rates for mean and covariance nets
    optimizer = optim.Adam([
        {'params': mean_net.parameters(), 'lr': 1e-5},       # higher LR for mean
        {'params': cov_net.parameters(), 'lr': 1e-5},      # lower LR for covariance
    ], weight_decay=wd)


    print(f"Training (full covariance) with output dim = {Y_len} and m = {Y_len*(Y_len+1)//2} tril params")
    best_val = float("inf")
    best_state = None

    mean_mats = [] 

    if train:

        train_losses, val_losses = [], []
        fig, ax = plt.subplots(figsize=(7,5))
        ax.set_title("Training vs Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        (line_train,) = ax.plot([], [], label="Train", lw=1.8)
        (line_val,)   = ax.plot([], [], label="Validation", lw=1.8)
        ax.legend(loc="best")
        plt.ion(); plt.show(block=False)

        for epoch in range(1, n_epochs + 1):
            tr_total, tr_mean, tr_cov = train_one_epoch(mean_net, cov_net, loss_fn, train_loader, optimizer)
            va_total, va_mean, va_cov = eval_epoch(mean_net, cov_net, loss_fn, valid_loader)  # <-- no trailing '\'

            M_epoch = epoch_mean_matrix(cov_net, valid_loader, device, which="Sigma")
            mean_mats.append(M_epoch)

            if epoch % 10 == 0 or epoch == 1:

                if va_total < best_val:
                    print(f"Epoch {epoch:04d} | Train: {tr_total:.6f} | mean: {tr_mean:.6f} | cov: {tr_cov:.6f} | "
                        f"Val: {va_total:.6f} | mean: {va_mean:.6f} | cov: {va_cov:.6f} (B)")
                else:
                    print(f"Epoch {epoch:04d} | Train: {tr_total:.6f} | mean: {tr_mean:.6f} | cov: {tr_cov:.6f} | "
                        f"Val: {va_total:.6f} | mean: {va_mean:.6f} | cov: {va_cov:.6f}")


            train_losses.append(tr_total)
            val_losses.append(va_total)

            # Update plot every N epochs (e.g. every 10)
            if epoch % 100 == 0:
                line_train.set_data(range(1, len(train_losses)+1), train_losses)
                line_val.set_data(range(1, len(val_losses)+1), val_losses)
                ax.relim(); ax.autoscale_view()
                plt.draw(); plt.pause(0.001)
                # Optionally save figure to file
                plt.savefig("/home/gjc7gx/CovNNet/Figures/ALLloss_curve.png", dpi=160)

            if va_total < best_val:
                print(f"Epoch {epoch:04d} | Train: {tr_total:.6f} | mean: {tr_mean:.6f} | cov: {tr_cov:.6f} | "
                    f"Val: {va_total:.6f} | mean: {va_mean:.6f} | cov: {va_cov:.6f} (B)")
                best_val = va_total
                best_state = (mean_net.state_dict(), cov_net.state_dict())

                mean_net.load_state_dict(best_state[0])
                cov_net.load_state_dict(best_state[1])

                meta = {
                    "Xn_key": Xn_key,                # normalization key for inputs
                    "Yn_key": Yn_key,                # normalization key for outputs (coeffs or bins)
                    "basis_meta": basis_meta,        # contains Phi1/Phi2, logspace flags, etc.
                    "r_tags": r_tags.tolist(),       # for plotting
                    "dims": {"X": X_len, "Y": Y_len}
                }
                save_checkpoint(CKPT_PATH, mean_net, cov_net, meta)

        # Plot covariance drifts (optional)
        if len(mean_mats) >= 2:
            titles = [f"Epoch {k+1}" for k in range(len(mean_mats))]
            plot_matrix_deltas(mean_mats, titles=titles)

    else:
        # No training: just load the existing checkpoint
        loaded_meta = load_checkpoint(CKPT_PATH, mean_net, cov_net, device)
        # Prefer keys/metadata from the checkpoint if present
        Yn_key     = loaded_meta.get("Yn_key", Yn_key)
        Xn_key     = loaded_meta.get("Xn_key", Xn_key)
        basis_meta = loaded_meta.get("basis_meta", basis_meta)
        if "r_tags" in loaded_meta:
            r_tags = np.array(loaded_meta["r_tags"], dtype=float)


    def _norm(vec_np, key):
        """
        Normalize a 1D vector using a group-based key.
        """
        assert vec_np.ndim == 1, "Only supports 1D vectors"
        vec_np = vec_np.astype(np.float32)
        result = np.empty_like(vec_np)
        
        for group in key["groups"]:
            cols = group["cols"]
            mode = group["mode"]
            lo = group["lo"]
            hi = group["hi"]
            shift = group["shift"]

            if mode == "linear":
                work = vec_np[cols]
            elif mode == "log":
                work = np.log(vec_np[cols] + shift + 1e-12)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            result[cols] = (work - lo) / (hi - lo)

        return result


    def _denorm(vec_np, key):
        """
        Denormalize a 1D vector using a group-based key.
        """
        assert vec_np.ndim == 1, "Only supports 1D vectors"
        vec_np = vec_np.astype(np.float32)
        result = np.empty_like(vec_np)

        for group in key["groups"]:
            cols = group["cols"]
            mode = group["mode"]
            lo = group["lo"]
            hi = group["hi"]
            shift = group["shift"]

            work = vec_np[cols] * (hi - lo) + lo

            if mode == "linear":
                result[cols] = work
            elif mode == "log":
                result[cols] = np.exp(work) - shift
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return result


    def _denorm_std(std_np, mu_n_np, key):
        """
        Approximate physical-scale std using groupwise Jacobian logic.
        """
        assert std_np.ndim == 1 and mu_n_np.ndim == 1
        std_np = std_np.astype(np.float32)
        mu_n_np = mu_n_np.astype(np.float32)
        result = np.empty_like(std_np)

        for group in key["groups"]:
            cols = group["cols"]
            mode = group["mode"]
            lo = group["lo"]
            hi = group["hi"]
            shift = group["shift"]
            scale = hi - lo

            if mode == "linear":
                result[cols] = std_np[cols] * scale
            elif mode == "log":
                work = mu_n_np[cols] * scale + lo
                result[cols] = std_np[cols] * np.exp(work) * scale
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return result


    def _linear_scales_from_key(key):
        """Return scale vector s (hi - lo) per output dim, for linear groups only."""
        s = np.zeros(sum(len(g["cols"]) for g in key["groups"]), dtype=np.float32)
        for g in key["groups"]:
            if g["mode"] != "linear":
                raise ValueError("Basis path expects coefficient groups to be 'linear'.")
            cols = g["cols"]; lo = g["lo"]; hi = g["hi"]
            s[np.array(cols, dtype=int)] = (hi - lo)
        return s  # [d]

    def _blockdiag(A, B):
        """Return blockdiag(A,B)."""
        m, n = A.shape; p, q = B.shape
        out = np.zeros((m+p, n+q), dtype=A.dtype)
        out[:m, :n] = A
        out[m:, n:] = B
        return out


    @torch.no_grad()
    def plot_one_simulation_with_bands_and_corr_blocks(
        mean_net,
        cov_net,
        x,                  # [dx]; normalized or raw
        y_true,             # [d];  normalized or raw (coeffs if using basis)
        r_tags,             # [Y1_len] radii (kpc)
        device,
        Y1_len=50,
        key_x=None,         # for X (only if x_is_normalized=False)
        key_y=None,         # for Y (coeff key if basis; bin key if raw)
        x_is_normalized=True,
        y_is_normalized=True,
        ci_sigma=2.0,       # band half-width in σ units
        n_r_ticks=6,        # how many radius ticks to show on heatmaps
        savepath="one_sim_blocks.png",
        title=None,
        basis_meta=None     # {"use_basis": bool, "y1":{"Phi":...}, "y2":{"Phi":...}}
    ):
        """
        Single-sample diagnostic plot:
        • Top row: Y1 (density) and Y2 (rotation) with ±ci_sigma bands.
            For Y1 the band is constructed in log-density space so the
            lower bound is strictly positive.
        • Bottom row: Corr(Y1,Y1), Corr(Y1,Y2), Corr(Y2,Y2) in bin space.
        """
        # ---- 1) prepare X ----
        x_np = np.asarray(x, dtype=np.float32)
        if not x_is_normalized:
            if key_x is None:
                raise ValueError("x_is_normalized=False but key_x is missing.")
            x_np = _norm(x_np, key_x)
        xb = torch.from_numpy(x_np).to(device).unsqueeze(0)  # [1, dx]

        # ---- 2) forward pass (normalized Y space) ----
        mean_net.eval()
        cov_net.eval()
        mu_n = mean_net(xb)                    # [1, d]
        L_n  = cov_net(xb)                     # [1, d, d]
        Sigma_n = L_n @ L_n.transpose(1, 2)    # [1, d, d]

        mu_n_np  = mu_n.squeeze(0).detach().cpu().numpy()    # [d]
        Sigma_np = Sigma_n.squeeze(0).detach().cpu().numpy() # [d, d]

        y_true_np = np.asarray(y_true, dtype=np.float32)

        use_basis = (basis_meta is not None) and basis_meta.get("use_basis", False)
        if not use_basis:
            raise ValueError(
                "This version of plot_one_simulation_with_bands_and_corr_blocks "
                "expects basis_meta['use_basis']=True (coeff → bins)."
            )
        if key_y is None:
            raise ValueError("basis path requires key_y for coefficient denorm.")

        # ------------------------------------------------------------------
        # 2a) BASIS PATH: coeffs → bins for both blocks
        # ------------------------------------------------------------------
        # Denormalize predicted & true coefficients (coeff space is linear-normalized)
        mu_coeff_phys = _denorm(mu_n_np, key_y)                # [d]
        y_coeff_phys  = _denorm(y_true_np, key_y) if y_is_normalized else y_true_np

        # Split coeff vectors by block
        k1 = int(basis_meta["y1"]["k"])
        c1_mu = mu_coeff_phys[:k1]
        c2_mu = mu_coeff_phys[k1:]
        c1_tr = y_coeff_phys[:k1]
        c2_tr = y_coeff_phys[k1:]

        # Build Phi and determine per-block mode (linear vs log)
        Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)  # [Y1_len, k1]
        Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)  # [Y2_len, k2]
        LOG1 = bool(basis_meta["y1"].get("logspace", False))
        LOG2 = bool(basis_meta["y2"].get("logspace", False))

        # Mean projection to bins
        lin_mu1 = Phi1 @ c1_mu
        lin_mu2 = Phi2 @ c2_mu
        mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
        mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2

        # True bins
        lin_tr1 = Phi1 @ c1_tr
        lin_tr2 = Phi2 @ c2_tr
        y1 = np.exp(lin_tr1) if LOG1 else lin_tr1
        y2 = np.exp(lin_tr2) if LOG2 else lin_tr2

        # ---- 2b) coefficient covariance → bin-space covariance ----
        # Coeff covariance: Σ_c = D Σ_n D, where D = diag(hi-lo) from coeff key (all linear)
        scales = _linear_scales_from_key(key_y).astype(np.float64)    # [d]
        D = np.diag(scales)                                           # [d,d]
        Sigma_c = D @ Sigma_np @ D.T                                  # [d,d]

        # Split coefficient covariance into blocks
        Sigma_c11 = Sigma_c[:k1, :k1]
        Sigma_c22 = Sigma_c[k1:, k1:]
        Sigma_c12 = Sigma_c[:k1, k1:]

        # Bin-space covariance for correlations (linear ρ and v)
        if LOG1:
            J1_rho = (mu1[:, None]) * Phi1   # d ρ / d c1
        else:
            J1_rho = Phi1
        if LOG2:
            J2 = (mu2[:, None]) * Phi2
        else:
            J2 = Phi2

        S11_rho = J1_rho @ Sigma_c11 @ J1_rho.T
        S22 = J2 @ Sigma_c22 @ J2.T
        S12 = J1_rho @ Sigma_c12 @ J2.T

        top = np.hstack([S11_rho, S12])
        bot = np.hstack([S12.T, S22])
        Sigma_bins = np.vstack([top, bot])   # [Y1_len+Y2_len, Y1_len+Y2_len]

        # Bin std for Y2 band and for correlations
        std_bins = np.sqrt(np.clip(np.diag(Sigma_bins), 0.0, None))
        std2 = std_bins[Y1_len:]

        # Correlation in bin space (still in linear ρ/v)
        stdn = np.sqrt(np.clip(np.diag(Sigma_bins), 1e-12, None))
        Corr = Sigma_bins / (stdn[:, None] * stdn[None, :] + 1e-12)
        C11 = Corr[:Y1_len, :Y1_len]
        C12 = Corr[:Y1_len, Y1_len:]
        C22 = Corr[Y1_len:, Y1_len:]

        # ---- 2c) LOG-SPACE covariance for Y1 bands (positive lower bound) ----
        if LOG1:
            J1_log = Phi1                     # d log ρ / d c1
            Sigma_log_y1 = J1_log @ Sigma_c11 @ J1_log.T
            var_log_y1 = np.clip(np.diag(Sigma_log_y1), 0.0, None)
            std_log_y1 = np.sqrt(var_log_y1)
            lower1 = np.exp(lin_mu1 - ci_sigma * std_log_y1)
            upper1 = np.exp(lin_mu1 + ci_sigma * std_log_y1)
        else:
            # fall back to symmetric linear band if not logspace
            std1 = std_bins[:Y1_len]
            lower1 = mu1 - ci_sigma * std1
            upper1 = mu1 + ci_sigma * std1

        # Y2 band (linear)
        lower2 = mu2 - ci_sigma * std2
        upper2 = mu2 + ci_sigma * std2

        # ---- 3) plotting ----
        r = np.asarray(r_tags, dtype=float)
        tick_idx = np.unique(np.linspace(0, Y1_len - 1, n_r_ticks).astype(int))
        tick_lbl = [f"{val:.2g}" for val in r[tick_idx]]

        fig = plt.figure(figsize=(14, 10))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.35)

        # Row 1: profiles
        top = outer[0].subgridspec(1, 2, wspace=0.25)
        ax11 = fig.add_subplot(top[0, 0])
        ax12 = fig.add_subplot(top[0, 1])

        # Density with band (log-space band mapped back to ρ)
        ax11.plot(r, y1,  label="True density", linewidth=1.5)
        ax11.plot(r, mu1, label="Pred mean",    linewidth=1.5)
        ax11.fill_between(r, lower1, upper1, alpha=0.25, label=f"±{ci_sigma}σ")
        ax11.set_xscale("log")
        ax11.set_yscale("log")
        ax11.set_ylim(1e2, 1e12)
        ax11.set_xlabel("Radius [kpc]")
        ax11.set_title("Y1: Density — True vs Pred (log-space band)")
        ax11.legend(loc="best")

        # Rotation with band (linear Gaussian)
        ax12.plot(r, y2,  label="True v_c", linewidth=1.5)
        ax12.plot(r, mu2, label="Pred mean", linewidth=1.5)
        ax12.fill_between(r, lower2, upper2, alpha=0.25, label=f"±{ci_sigma}σ")
        ax12.set_xscale("log")
        ax12.set_xlabel("Radius [kpc]")
        ax12.set_ylim(0, 500)
        ax12.set_title("Y2: Rotation — True vs Pred (± band)")
        ax12.legend(loc="best")

        # Row 2: correlation heatmaps
        bot = outer[1].subgridspec(1, 3, wspace=0.25)
        ax21 = fig.add_subplot(bot[0, 0])
        ax22 = fig.add_subplot(bot[0, 1])
        ax23 = fig.add_subplot(bot[0, 2])

        im1 = ax21.imshow(C11, vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")
        im2 = ax22.imshow(C12, vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")
        im3 = ax23.imshow(C22, vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")

        for ax, ttl, xlbl, ylbl in [
            (ax21, "Corr(Y1, Y1)", "Radius [kpc]", "Radius [kpc]"),
            (ax22, "Corr(Y1, Y2)", "Y2 radius [kpc]", "Y1 radius [kpc]"),
            (ax23, "Corr(Y2, Y2)", "Radius [kpc]", "Radius [kpc]"),
        ]:
            ax.set_title(ttl)
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_lbl, rotation=45, ha="right")
            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_lbl)
            ax.invert_yaxis()

        cbar = fig.colorbar(im3, ax=[ax21, ax22, ax23], fraction=0.046, pad=0.04)
        cbar.set_label("corr")

        if title:
            fig.suptitle(title, y=0.995, fontsize=14)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=160)
            print(f"Saved: ", savepath)
        return fig


    from matplotlib.animation import FuncAnimation, PillowWriter

    @torch.no_grad()
    def animate_sweep_one_feature(
        mean_net,
        cov_net,
        train_loader,         # used to estimate normalized feature means
        sweep_idx: int,       # which X-dimension to sweep (0-based)
        r_tags,               # [Y1_len] radii (kpc)
        device,
        Y1_len=50,
        key_x=None,           # not needed if x is already normalized in loader
        key_y=None,           # coeff key if using basis
        x_is_normalized=True,
        ci_sigma=2.0,
        n_r_ticks=6,
        basis_meta=None,      # {"use_basis": bool, "y1":{"Phi","k","logspace"}, "y2":{...}}
        sweep_vals=None,      # iterable of normalized values to use for the selected feature
        fps=20,
        savepath_gif="sweep_x.gif",
        title_prefix=None
    ):
        """
        Animate predicted Density/Rotation bands and correlation blocks
        while sweeping a single normalized input feature X[sweep_idx].

        Assumes your model outputs coefficient space when basis_meta['use_basis']=True.
        """
        use_basis = (basis_meta is not None) and bool(basis_meta.get("use_basis", False))
        if not use_basis:
            raise ValueError(
                "animate_sweep_one_feature currently targets the basis path "
                "(coeff → bins). Provide basis_meta with use_basis=True."
            )
        if key_y is None:
            raise ValueError("basis path requires key_y for coefficient denorm.")

        # ------------------------ 1) estimate normalized X mean ------------------------
        n_seen, x_sum = 0, None
        for xb, _ in train_loader:
            xb = xb.to(device)
            if x_sum is None:
                x_sum = xb.sum(dim=0)
            else:
                x_sum += xb.sum(dim=0)
            n_seen += xb.shape[0]
        X_ref = (x_sum / max(1, n_seen)).detach().cpu().numpy().astype(np.float32)

        dx = X_ref.shape[0]
        if sweep_idx < 0 or sweep_idx >= dx:
            raise ValueError(f"sweep_idx={sweep_idx} out of range for X dimension {dx}")

        if sweep_vals is None:
            sweep_vals = np.linspace(0.05, 0.95, 121, dtype=np.float32)

        # ------------------------ 2) set up basis & figure ------------------------
        k1 = int(basis_meta["y1"]["k"])
        Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)
        Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)
        LOG1 = bool(basis_meta["y1"].get("logspace", False))
        LOG2 = bool(basis_meta["y2"].get("logspace", False))

        r = np.asarray(r_tags, dtype=float)
        tick_idx = np.unique(np.linspace(0, Y1_len - 1, n_r_ticks).astype(int))
        tick_lbl = [f"{val:.2g}" for val in r[tick_idx]]

        fig = plt.figure(figsize=(14, 10))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.35)
        top = outer[0].subgridspec(1, 2, wspace=0.25)
        bot = outer[1].subgridspec(1, 3, wspace=0.25)

        ax11 = fig.add_subplot(top[0, 0])   # Density
        ax12 = fig.add_subplot(top[0, 1])   # Rotation
        ax21 = fig.add_subplot(bot[0, 0])   # Corr(Y1,Y1)
        ax22 = fig.add_subplot(bot[0, 1])   # Corr(Y1,Y2)
        ax23 = fig.add_subplot(bot[0, 2])   # Corr(Y2,Y2)

        # Static axis configs
        ax11.set_xscale("log")
        ax11.set_yscale("log")
        ax11.set_xlabel("Radius [kpc]")
        ax11.set_title("Y1: Density — Pred (log-space band)")
        ax11.set_ylim(1e3, 1e10)

        ax12.set_xscale("log")
        ax12.set_xlabel("Radius [kpc]")
        ax12.set_ylim(0, 500)
        ax12.set_title("Y2: Rotation — Pred (± band)")

        for ax, ttl, xlbl, ylbl in [
            (ax21, "Corr(Y1, Y1)", "Radius [kpc]", "Radius [kpc]"),
            (ax22, "Corr(Y1, Y2)", "Y2 radius [kpc]", "Y1 radius [kpc]"),
            (ax23, "Corr(Y2, Y2)", "Radius [kpc]", "Radius [kpc]"),
        ]:
            ax.set_title(ttl)
            ax.set_xlabel(xlbl)
            ax.set_ylabel(ylbl)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_lbl, rotation=45, ha="right")
            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_lbl)
            ax.invert_yaxis()

        # Artists to update
        (line_y1,) = ax11.plot(r, np.ones_like(r), linewidth=1.8, label="Pred mean")
        (line_y2,) = ax12.plot(r, np.ones_like(r), linewidth=1.8, label="Pred mean")
        band1 = ax11.fill_between(r, r * 0, r * 0, alpha=0.25,
                                label=f"±{ci_sigma}σ", color="cornflowerblue")
        band2 = ax12.fill_between(r, r * 0, r * 0, alpha=0.25,
                                label=f"±{ci_sigma}σ", color="cornflowerblue")
        ax11.legend(loc="best")
        ax12.legend(loc="best")

        im1 = ax21.imshow(np.zeros((Y1_len, Y1_len)), vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")
        im2 = ax22.imshow(np.zeros((Y1_len, Y1_len)), vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")
        im3 = ax23.imshow(np.zeros((Y1_len, Y1_len)), vmin=-1, vmax=1, cmap="coolwarm",
                        origin="upper", aspect="auto")
        cbar = fig.colorbar(im3, ax=[ax21, ax22, ax23], fraction=0.046, pad=0.04)
        cbar.set_label("corr")

        # Optional title that updates with the swept value
        if title_prefix is None:
            title_prefix = f"Sweep X[{sweep_idx}]"
        title_text = fig.suptitle(f"{title_prefix}", y=0.995, fontsize=14)

        # Coeff scales for covariance transform
        scales_c = _linear_scales_from_key(key_y).astype(np.float64)

        # ------------------------ 3) frame computation ------------------------
        def compute_pred_bins(xn_vec: np.ndarray):
            xb = torch.from_numpy(xn_vec.astype(np.float32)).to(device).unsqueeze(0)  # [1,dx]
            mean_net.eval()
            cov_net.eval()
            mu_n = mean_net(xb)                    # [1,d]
            L_n  = cov_net(xb)                     # [1,d,d]
            Sigma_n = L_n @ L_n.transpose(1, 2)    # [1,d,d]
            mu_n_np  = mu_n.squeeze(0).detach().cpu().numpy()
            Sigma_np = Sigma_n.squeeze(0).detach().cpu().numpy()

            # coeff denorm (linear groups)
            c_mu = _denorm(mu_n_np, key_y)
            c1_mu, c2_mu = c_mu[:k1], c_mu[k1:]

            # mean in bins
            lin_mu1 = Phi1 @ c1_mu
            lin_mu2 = Phi2 @ c2_mu
            mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
            mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2

            # Coeff covariance: Σ_c = D Σ_n D
            D = np.diag(scales_c)
            Sigma_c = D @ Sigma_np @ D.T

            Sigma_c11 = Sigma_c[:k1, :k1]
            Sigma_c22 = Sigma_c[k1:, k1:]
            Sigma_c12 = Sigma_c[:k1, k1:]

            # Bin-space covariance for correlations (linear ρ/v)
            if LOG1:
                J1_rho = (mu1[:, None]) * Phi1
            else:
                J1_rho = Phi1
            if LOG2:
                J2 = (mu2[:, None]) * Phi2
            else:
                J2 = Phi2

            S11_rho = J1_rho @ Sigma_c11 @ J1_rho.T
            S22 = J2 @ Sigma_c22 @ J2.T
            S12 = J1_rho @ Sigma_c12 @ J2.T

            top = np.hstack([S11_rho, S12])
            bot = np.hstack([S12.T, S22])
            Sigma_bins = np.vstack([top, bot])

            # std for Y2 band
            std_bins = np.sqrt(np.clip(np.diag(Sigma_bins), 0.0, None))
            std2 = std_bins[Y1_len:]

            # correlations
            stdn = np.sqrt(np.clip(np.diag(Sigma_bins), 1e-12, None))
            Corr = Sigma_bins / (stdn[:, None] * stdn[None, :] + 1e-12)
            C11 = Corr[:Y1_len, :Y1_len]
            C12 = Corr[:Y1_len, Y1_len:]
            C22 = Corr[Y1_len:, Y1_len:]

            # log-space band for Y1
            if LOG1:
                J1_log = Phi1
                Sigma_log_y1 = J1_log @ Sigma_c11 @ J1_log.T
                var_log_y1 = np.clip(np.diag(Sigma_log_y1), 0.0, None)
                std_log_y1 = np.sqrt(var_log_y1)
                lower1 = np.exp(lin_mu1 - ci_sigma * std_log_y1)
                upper1 = np.exp(lin_mu1 + ci_sigma * std_log_y1)
            else:
                std1 = std_bins[:Y1_len]
                lower1 = mu1 - ci_sigma * std1
                upper1 = mu1 + ci_sigma * std1

            lower2 = mu2 - ci_sigma * std2
            upper2 = mu2 + ci_sigma * std2

            return mu1, mu2, lower1, upper1, lower2, upper2, C11, C12, C22

        # ------------------------ 4) animation update ------------------------
        patches = {"band1": band1, "band2": band2}

        def update(frame_idx):
            xn = X_ref.copy()
            xn[sweep_idx] = float(sweep_vals[frame_idx])
            mu1, mu2, lower1, upper1, lower2, upper2, C11, C12, C22 = compute_pred_bins(xn)

            # lines
            line_y1.set_ydata(mu1)
            line_y2.set_ydata(mu2)

            # replace shaded bands each frame
            if patches["band1"] is not None:
                patches["band1"].remove()
            if patches["band2"] is not None:
                patches["band2"].remove()
            patches["band1"] = ax11.fill_between(
                r, lower1, upper1, alpha=0.25, color="cornflowerblue"
            )
            patches["band2"] = ax12.fill_between(
                r, lower2, upper2, alpha=0.25, color="cornflowerblue"
            )

            # heatmaps
            im1.set_data(C11)
            im2.set_data(C12)
            im3.set_data(C22)

            # Update title with current swept value
            title_text.set_text(f"{title_prefix} = {xn[sweep_idx]:.3f} (normalized)")

            return (line_y1, line_y2, patches["band1"], patches["band2"],
                    im1, im2, im3, title_text)

        ani = FuncAnimation(fig, update, frames=len(sweep_vals),
                            blit=False, interval=1000.0 / fps)

        writer = PillowWriter(fps=fps)
        ani.save(savepath_gif, writer=writer, dpi=160)
        print(f"[anim] Saved GIF to: {savepath_gif}")

        return ani


    xb, yb = next(iter(test_loader))
    _ = plot_one_simulation_with_bands_and_corr_blocks(
            mean_net, cov_net,
            xb[0].cpu().numpy(), yb[0].cpu().numpy(),
            r_tags, device,
            Y1_len=50,
            key_x=Xn_key, key_y=Yn_key,           # coeff key if using basis
            x_is_normalized=True, y_is_normalized=True,
            ci_sigma=2.0,
            savepath="/home/gjc7gx/CovNNet/Figures/ALL_one_sim_with_bands.png",
            basis_meta=basis_meta                 # <- pass your basis meta
    )

    sweep_idx = 2
    sweep_vals = np.linspace(0.05, 0.95, 121)  # normalized

    _ = animate_sweep_one_feature(
            mean_net, cov_net,
            train_loader=train_loader,
            sweep_idx=sweep_idx,
            r_tags=r_tags,
            device=device,
            Y1_len=50,
            key_x=Xn_key,
            key_y=Yn_key,
            x_is_normalized=True,
            ci_sigma=2.0,
            n_r_ticks=6,
            basis_meta=basis_meta,
            sweep_vals=sweep_vals,
            fps=20,
            savepath_gif=f"/home/gjc7gx/CovNNet/Figures/sweep_x{sweep_idx}.gif",
            title_prefix=f"Sweep X[{sweep_idx}]"
    )
