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
import numpy as np
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

    train = False
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

    # ----------------------- Switch: raw vs basis -----------------------
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
        Yn, Yn_key = norm_params(Y, Y_groups, margin_pct=0.05)

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
        Yn, Yn_key = norm_params(Y, Y_groups, margin_pct=0.05)
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
    MODEL_NAME = f"test3"
    CKPT_PATH  = os.path.join(MODEL_DIR, MODEL_NAME + ".pt")
    
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
                    min_std=1e-4, max_std=None, off_cap=0.3, off_warmup_scale=1.0,
                    diag_temp=1.0, **unused):
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



    def probe_L_diff(L, diff, tag=""):
        diag = torch.diagonal(L, dim1=1, dim2=2)                 # [B,d]
        min_abs_diag = diag.abs().min().item()
        any_bad_diag = (~torch.isfinite(diag)).any().item() or (min_abs_diag < 1e-8)
        print(f"[{tag}] "
            f"L finite? {torch.isfinite(L).all().item()} | "
            f"diff finite? {torch.isfinite(diff).all().item()} | "
            f"min(|diag(L)|)={min_abs_diag:.3e} | "
            f"max|L|={L.abs().max().item():.3e} | "
            f"max|diff|={diff.abs().max().item():.3e} | "
            f"bad_diag? {any_bad_diag}")

    import torch; torch.autograd.set_detect_anomaly(True)


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


    def epoch_cov_stats(cov_net, loader, device):
        diag_mins, diag_meds, diag_maxs = [], [], []
        off_norms = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                L = cov_net(xb)                            # [B,d,d]
                diag = torch.diagonal(L, dim1=1, dim2=2)   # [B,d]
                off  = L - torch.diag_embed(diag)

                diag_mins.append(diag.min().item())
                diag_meds.append(diag.median().item())
                diag_maxs.append(diag.max().item())
                off_norms.append(off.pow(2).sum(dim=2).sqrt().mean().item())

        # summarize across batches
        print(
            f"diag std min/med/max (epoch): "
            f"{np.mean(diag_mins):.3e} / {np.mean(diag_meds):.3e} / {np.mean(diag_maxs):.3e}"
        )
        print(f"off Frobenius row-norm (mean, epoch): {np.mean(off_norms):.3e}")


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

    # -------------------------
    # Hyperparameters & build
    # -------------------------
    FULL_COV = True  # keep True to learn full correlation structure
    diag_eps = 1e-6
    cov_hc       = 64   # hidden width
    cov_nl       = 2     # hidden layers
    mean_hc       = 256   # hidden width
    mean_nl       = 3     # hidden layers
    lr       = 0.5e-5
    wd       = 1e-5
    n_epochs = 25000

    mean_net, cov_net, loss_fn = make_models(
        in_dim=X.shape[1], out_dim=Y_len, cov_hc=cov_hc, cov_nl=cov_nl, mean_hc=mean_hc, mean_nl=mean_nl,diag_eps=diag_eps
    )
    optimizer = optim.Adam(list(mean_net.parameters()) + list(cov_net.parameters()),
                        lr=lr, weight_decay=wd)

    print(f"Training (full covariance) with output dim = {Y_len} and m = {Y_len*(Y_len+1)//2} tril params")
    best_val = float("inf")
    best_state = None

    mean_mats = [] 

    if train:
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

            if va_total < best_val:
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
        basis_meta=None     # NEW: {"use_basis": bool, "y1":{"Phi":...}, "y2":{"Phi":...}}
    ):
        # ---- prep X ----
        x_np = np.asarray(x, dtype=np.float32)
        if not x_is_normalized:
            if key_x is None:
                raise ValueError("x_is_normalized=False but key_x is missing.")
            x_np = _norm(x_np, key_x)
        xb = torch.from_numpy(x_np).to(device).unsqueeze(0)

        # ---- forward (normalized Y) ----
        mean_net.eval(); cov_net.eval()
        mu_n = mean_net(xb)                        # [1,d]
        L_n  = cov_net(xb)                         # [1,d,d]
        Sigma_n = L_n @ L_n.transpose(1,2)         # [1,d,d]

        mu_n_np  = mu_n.squeeze(0).cpu().numpy()         # [d]
        Sigma_np = Sigma_n.squeeze(0).cpu().numpy()      # [d,d]

        # ---- choose plotting space ----
        y_true_np = np.asarray(y_true, dtype=np.float32)
        use_basis = (basis_meta is not None) and basis_meta.get("use_basis", False)

        if use_basis:
            if key_y is None:
                raise ValueError("basis path requires key_y for coefficient denorm.")

            # ---- 2.1: denormalize predicted & true coefficients (coeff space is linear-normalized) ----
            mu_coeff_phys = _denorm(mu_n_np, key_y)                # [d]
            y_coeff_phys  = _denorm(y_true_np, key_y) if y_is_normalized else y_true_np

            # Split coeff vectors by block
            k1 = basis_meta["y1"]["k"]
            c1_mu = mu_coeff_phys[:k1]
            c2_mu = mu_coeff_phys[k1:]
            c1_tr = y_coeff_phys[:k1]
            c2_tr = y_coeff_phys[k1:]

            # ---- 2.2: build Phi and determine per-block mode (linear vs log) ----
            Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)  # [50, k1]
            Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)  # [50, k2]
            LOG1 = bool(basis_meta["y1"].get("logspace", False))
            LOG2 = bool(basis_meta["y2"].get("logspace", False))

            # ---- 2.3: mean projection to bins ----
            # linear block: y = Phi @ c
            # log block   : y = exp(Phi @ c)
            lin_mu1 = Phi1 @ c1_mu
            lin_mu2 = Phi2 @ c2_mu
            mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
            mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2

            # true bins from coeffs (for plotting ground truth)
            lin_tr1 = Phi1 @ c1_tr
            lin_tr2 = Phi2 @ c2_tr
            y1 = np.exp(lin_tr1) if LOG1 else lin_tr1
            y2 = np.exp(lin_tr2) if LOG2 else lin_tr2

            # ---- 2.4: covariance projection with Jacobians ----
            # Coeff covariance: Σ_c = D Σ_n D, where D = diag(hi-lo) from coeff key (linear groups)
            scales = _linear_scales_from_key(key_y).astype(np.float64)    # [d]
            D = np.diag(scales)                                           # [d,d]
            Sigma_c = D @ Sigma_np @ D.T                                  # [d,d]

            # Build per-block Jacobians J1, J2 at the predicted mean:
            # linear block: y = Phi c             -> J = Phi
            # log block   : y = exp(Phi c)       -> J = diag(mu_bin) @ Phi
            if LOG1:
                J1 = (mu1[:, None]) * Phi1       # [50, k1]
            else:
                J1 = Phi1

            if LOG2:
                J2 = (mu2[:, None]) * Phi2       # [50, k2]
            else:
                J2 = Phi2

            # Combine into block-Jacobian and block-split Sigma_c
            Sigma_c11 = Sigma_c[:k1, :k1]
            Sigma_c22 = Sigma_c[k1:, k1:]
            Sigma_c12 = Sigma_c[:k1, k1:]

            # Bin-space covariance blocks
            S11 = J1 @ Sigma_c11 @ J1.T
            S22 = J2 @ Sigma_c22 @ J2.T
            S12 = J1 @ Sigma_c12 @ J2.T
            # stitch full [100x100]
            top = np.hstack([S11, S12])
            bot = np.hstack([S12.T, S22])
            Sigma_bins = np.vstack([top, bot])

            # Bin std (for shaded bands)
            std_bins_phys = np.sqrt(np.clip(np.diag(Sigma_bins), 0.0, None))
            std1, std2 = std_bins_phys[:Y1_len], std_bins_phys[Y1_len:]

            # correlation in bin space
            std_n_bins = np.sqrt(np.clip(np.diag(Sigma_bins), 1e-12, None))
            Corr = Sigma_bins / (std_n_bins[:, None] * std_n_bins[None, :] + 1e-12)
            C11 = Corr[:Y1_len, :Y1_len]
            C12 = Corr[:Y1_len, Y1_len:]
            C22 = Corr[Y1_len:, Y1_len:]

        # tick helpers (use radii instead of indices)
        r = np.asarray(r_tags, dtype=float)
        tick_idx = np.unique(np.linspace(0, Y1_len-1, n_r_ticks).astype(int))
        tick_lbl = [f"{val:.2g}" for val in r[tick_idx]]

        # ---- layout: row1 two plots; row2 three heatmaps ----
        fig = plt.figure(figsize=(14, 10))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.35)

        # Row 1
        top = outer[0].subgridspec(1, 2, wspace=0.25)
        ax11 = fig.add_subplot(top[0, 0])
        ax12 = fig.add_subplot(top[0, 1])

        # Density with band
        ax11.plot(r, y1,  label="True density", linewidth=1.5)
        ax11.plot(r, mu1, label="Pred mean",    linewidth=1.5)
        ax11.fill_between(r, mu1 - ci_sigma*std1, mu1 + ci_sigma*std1, alpha=0.25, label=f"±{ci_sigma}σ")
        ax11.set_xscale("log")
        ax11.set_yscale("log")
        ax11.set_xlabel("Radius [kpc]")
        ax11.set_title("Y1: Density — True vs Pred (± band)")
        ax11.legend(loc="best")

        # Rotation with band
        ax12.plot(r, y2,  label="True v_c", linewidth=1.5)
        ax12.plot(r, mu2, label="Pred mean", linewidth=1.5)
        ax12.fill_between(r, mu2 - ci_sigma*std2, mu2 + ci_sigma*std2, alpha=0.25, label=f"±{ci_sigma}σ")
        ax12.set_xscale("log"); ax12.set_xlabel("Radius [kpc]")
        ax12.set_ylim(0,600)
        ax12.set_title("Y2: Rotation — True vs Pred (± band)")
        ax12.legend(loc="best")

        # Row 2 (three correlation blocks)
        bot = outer[1].subgridspec(1, 3, wspace=0.25)
        ax21 = fig.add_subplot(bot[0, 0])  # C11
        ax22 = fig.add_subplot(bot[0, 1])  # C12
        ax23 = fig.add_subplot(bot[0, 2])  # C22

        im1 = ax21.imshow(C11, vmin=-1, vmax=1, cmap="coolwarm", origin="upper", aspect="auto")
        ax21.set_title("Corr(Y1, Y1)")
        ax21.set_xlabel("Radius [kpc]"); ax21.set_ylabel("Radius [kpc]")
        ax21.set_xticks(tick_idx); ax21.set_xticklabels(tick_lbl, rotation=45, ha="right")
        ax21.set_yticks(tick_idx); ax21.set_yticklabels(tick_lbl)
        ax21.invert_yaxis()

        im2 = ax22.imshow(C12, vmin=-1, vmax=1, cmap="coolwarm", origin="upper", aspect="auto")
        ax22.set_title("Corr(Y1, Y2)")
        ax22.set_xlabel("Y2 radius [kpc]"); ax22.set_ylabel("Y1 radius [kpc]")
        ax22.set_xticks(tick_idx); ax22.set_xticklabels(tick_lbl, rotation=45, ha="right")
        ax22.set_yticks(tick_idx); ax22.set_yticklabels(tick_lbl)
        ax22.invert_yaxis()

        im3 = ax23.imshow(C22, vmin=-1, vmax=1, cmap="coolwarm", origin="upper", aspect="auto")
        ax23.set_title("Corr(Y2, Y2)")
        ax23.set_xlabel("Radius [kpc]"); ax23.set_ylabel("Radius [kpc]")
        ax23.set_xticks(tick_idx); ax23.set_xticklabels(tick_lbl, rotation=45, ha="right")
        ax23.set_yticks(tick_idx); ax23.set_yticklabels(tick_lbl)
        ax23.invert_yaxis()

        # single shared colorbar for all three
        cbar = fig.colorbar(im3, ax=[ax21, ax22, ax23], fraction=0.046, pad=0.04)
        cbar.set_label("corr")

        if title:
            fig.suptitle(title, y=0.995, fontsize=14)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=160)
            print(f"Saved: {savepath}")
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
        Animate predicted Density/Rotation ±σ bands and correlation blocks
        while sweeping a single normalized input feature X[sweep_idx].

        Assumes your model outputs coefficient space when basis_meta['use_basis']=True.
        """

        # ------------------------ 0) helpers ------------------------
        # If you rely on this helper elsewhere it’s fine to remove here.
        def _linear_scales_from_key_local(key):
            # builds scale (hi - lo) per column for *linear* groups in key
            dim = key.get("dim", None)
            if dim is None:
                # infer max col index
                m = 0
                for g in key["groups"]:
                    m = max(m, int(np.max(g["cols"])) + 1)
                dim = m
            scales = np.ones(dim, dtype=np.float64)
            for g in key["groups"]:
                cols = np.asarray(g["cols"], dtype=int)
                lo, hi = np.asarray(g["lo"]), np.asarray(g["hi"])
                # Support scalar or per-col lo/hi
                sc = (hi - lo).astype(np.float64) if np.ndim(hi) else float(hi - lo)
                scales[cols] = sc
            return scales

        # pick helper (use your existing one if present)
        _scales_from_key = _linear_scales_from_key if ' _linear_scales_from_key' in globals() else _linear_scales_from_key_local

        # ------------------------ 1) estimate normalized X mean ------------------------
        # Compute normalized mean from the loader (works even if batches are shuffled)
        n_seen, x_sum = 0, None
        for xb, _ in train_loader:
            xb = xb.to(device)
            if x_sum is None:
                x_sum = xb.sum(dim=0)
            else:
                x_sum += xb.sum(dim=0)
            n_seen += xb.shape[0]
        X_ref = (x_sum / max(1, n_seen)).detach().cpu().numpy().astype(np.float32)  # normalized reference vector

        dx = X_ref.shape[0]
        if sweep_idx < 0 or sweep_idx >= dx:
            raise IndexError(f"sweep_idx={sweep_idx} out of range for X-dim={dx}")

        # Values to sweep (in normalized space). Default: 0.05→0.95 for safety.
        if sweep_vals is None:
            sweep_vals = np.linspace(0.05, 0.95, 121, dtype=np.float32)

        # ------------------------ 2) set up figure / axes ------------------------
        use_basis = (basis_meta is not None) and bool(basis_meta.get("use_basis", False))
        if not use_basis:
            raise ValueError("This animator currently targets the basis path (coeff → bins). Provide basis_meta with use_basis=True.")

        k1 = int(basis_meta["y1"]["k"])
        Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)
        Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)
        LOG1 = bool(basis_meta["y1"].get("logspace", False))
        LOG2 = bool(basis_meta["y2"].get("logspace", False))

        # tick helpers based on radii
        r = np.asarray(r_tags, dtype=float)
        tick_idx = np.unique(np.linspace(0, Y1_len-1, n_r_ticks).astype(int))
        tick_lbl = [f"{val:.2g}" for val in r[tick_idx]]

        # figure layout: same as your function, but no trues
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
        ax11.set_xscale("log"); ax11.set_yscale("log"); ax11.set_xlabel("Radius [kpc]")
        ax11.set_title("Y1: Density — Pred (± band)")
        ax11.set_ylim(1e+3, 1e+10)

        ax12.set_xscale("log"); ax12.set_xlabel("Radius [kpc]")
        ax12.set_ylim(0, 500)
        ax12.set_title("Y2: Rotation — Pred (± band)")

        for ax, ttl, xlbl, ylbl in [
            (ax21, "Corr(Y1, Y1)", "Radius [kpc]", "Radius [kpc]"),
            (ax22, "Corr(Y1, Y2)", "Y2 radius [kpc]", "Y1 radius [kpc]"),
            (ax23, "Corr(Y2, Y2)", "Radius [kpc]", "Radius [kpc]"),
        ]:
            ax.set_title(ttl)
            ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
            ax.set_xticks(tick_idx); ax.set_xticklabels(tick_lbl, rotation=45, ha="right")
            ax.set_yticks(tick_idx); ax.set_yticklabels(tick_lbl)
            ax.invert_yaxis()

        # Artists to update
        (line_y1,) = ax11.plot(r, np.ones_like(r), linewidth=1.8, label="Pred mean")
        (line_y2,) = ax12.plot(r, np.ones_like(r), linewidth=1.8, label="Pred mean")
        band1 = ax11.fill_between(r, r*0, r*0, alpha=0.25, label=f"±{ci_sigma}σ", color="cornflowerblue")
        band2 = ax12.fill_between(r, r*0, r*0, alpha=0.25, label=f"±{ci_sigma}σ", color="cornflowerblue")
        ax11.legend(loc="best"); ax12.legend(loc="best")

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

        # ------------------------ 3) frame computation ------------------------
        scales_c = _scales_from_key(key_y).astype(np.float64)  # for coeff covariance transform

        def compute_pred_bins(xn_vec: np.ndarray):
            xb = torch.from_numpy(xn_vec.astype(np.float32)).to(device).unsqueeze(0)  # [1,dx]
            mean_net.eval(); cov_net.eval()
            mu_n = mean_net(xb)                    # [1,d]
            L_n  = cov_net(xb)                     # [1,d,d]
            Sigma_n = L_n @ L_n.transpose(1, 2)    # [1,d,d]
            mu_n_np  = mu_n.squeeze(0).cpu().numpy()
            Sigma_np = Sigma_n.squeeze(0).cpu().numpy()

            # coeff denorm (linear groups): c_phys = lo + n * (hi-lo)  (handled by your _denorm)
            c_mu = _denorm(mu_n_np, key_y)
            # Project to bins
            c1_mu, c2_mu = c_mu[:k1], c_mu[k1:]
            lin_mu1 = Phi1 @ c1_mu
            lin_mu2 = Phi2 @ c2_mu
            mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
            mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2

            # Covariance projection: Σ_c = D Σ_n D, D = diag(hi-lo)
            D = np.diag(scales_c)
            Sigma_c = D @ Sigma_np @ D.T

            # Jacobians for bin space at mean
            J1 = (mu1[:, None]) * Phi1 if LOG1 else Phi1
            J2 = (mu2[:, None]) * Phi2 if LOG2 else Phi2

            Sigma_c11 = Sigma_c[:k1, :k1]
            Sigma_c22 = Sigma_c[k1:, k1:]
            Sigma_c12 = Sigma_c[:k1, k1:]

            S11 = J1 @ Sigma_c11 @ J1.T
            S22 = J2 @ Sigma_c22 @ J2.T
            S12 = J1 @ Sigma_c12 @ J2.T

            top = np.hstack([S11, S12])
            bot = np.hstack([S12.T, S22])
            Sigma_bins = np.vstack([top, bot])

            std_bins = np.sqrt(np.clip(np.diag(Sigma_bins), 0.0, None))
            std1, std2 = std_bins[:Y1_len], std_bins[Y1_len:]

            # correlation blocks
            stdn = np.sqrt(np.clip(np.diag(Sigma_bins), 1e-12, None))
            Corr = Sigma_bins / (stdn[:, None] * stdn[None, :] + 1e-12)
            C11 = Corr[:Y1_len, :Y1_len]
            C12 = Corr[:Y1_len, Y1_len:]
            C22 = Corr[Y1_len:, Y1_len:]

            return mu1, mu2, std1, std2, C11, C12, C22

        # ------------------------ 4) animation update ------------------------
        patches = {"band1": band1, "band2": band2}  # to remove/replace bands cleanly

        def update(frame_idx):
            xn = X_ref.copy()
            xn[sweep_idx] = float(sweep_vals[frame_idx])
            mu1, mu2, std1, std2, C11, C12, C22 = compute_pred_bins(xn)

            # lines
            line_y1.set_ydata(mu1)
            line_y2.set_ydata(mu2)

            # replace shaded bands each frame (simple & robust)
            if patches["band1"] is not None:
                patches["band1"].remove()
            if patches["band2"] is not None:
                patches["band2"].remove()
            patches["band1"] = ax11.fill_between(r, mu1 - ci_sigma*std1, mu1 + ci_sigma*std1, alpha=0.25, color="cornflowerblue")
            patches["band2"] = ax12.fill_between(r, mu2 - ci_sigma*std2, mu2 + ci_sigma*std2, alpha=0.25, color="cornflowerblue")

            # heatmaps
            im1.set_data(C11)
            im2.set_data(C12)
            im3.set_data(C22)

            # title with current value (normalized)
            title_text.set_text(f"{title_prefix} = {xn[sweep_idx]:.3f} (normalized)")

            # return updated artists
            return (line_y1, line_y2, patches["band1"], patches["band2"], im1, im2, im3, title_text)

        ani = FuncAnimation(fig, update, frames=len(sweep_vals), blit=False, interval=1000/fps)

        # Save GIF
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
            savepath="/home/gjc7gx/CovNNet/Figures/one_sim_with_bands.png",
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

    # --- Helpers to project a normalized X vector to bin-space profiles ---
    @torch.no_grad()
    def _predict_bins_from_xn(mean_net, xn_vec, device, key_y, basis_meta):
        """
        xn_vec: normalized input vector (shape [dx])
        Returns: (r, mu1, mu2) where mu1 = density profile, mu2 = rotation profile
        """
        mean_net.eval()
        xb = torch.from_numpy(xn_vec.astype(np.float32)).to(device).unsqueeze(0)  # [1,dx]
        mu_n = mean_net(xb)                        # [1,d_y]
        mu_n_np = mu_n.squeeze(0).cpu().numpy()    # [d_y]

        # Denormalize coefficients and project to bins
        mu_coeff_phys = _denorm(mu_n_np, key_y)

        k1 = int(basis_meta["y1"]["k"])
        Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)
        Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)
        LOG1 = bool(basis_meta["y1"].get("logspace", False))
        LOG2 = bool(basis_meta["y2"].get("logspace", False))

        c1_mu, c2_mu = mu_coeff_phys[:k1], mu_coeff_phys[k1:]
        lin_mu1 = Phi1 @ c1_mu
        lin_mu2 = Phi2 @ c2_mu
        mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
        mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2
        return mu1, mu2


    # --- Export predicted profile for the MEAN input X (unchanged behavior) ---
    @torch.no_grad()
    def export_profile_mean_input(mean_net, cov_net, Xn, r_tags, device,
                                key_x=None, key_y=None, basis_meta=None,
                                savepath="predicted_profile_mean_input.txt"):
        """
        Exports the predicted mean density (mu1) and rotation (mu2) profiles
        for the mean input vector across the dataset.
        """
        # Compute normalized mean input
        x_mean = Xn.mean(axis=0).astype(np.float32)  # already normalized

        # Forward pass → bin-space
        mu1, mu2 = _predict_bins_from_xn(mean_net, x_mean, device, key_y, basis_meta)

        # Stack results [radius, mu1, mu2]
        out = np.column_stack([r_tags, mu1, mu2])
        np.savetxt(savepath, out, header="radius_kpc  density_pred  v_c_pred")
        print(f"[export] Saved mean-input predicted profile to {savepath}")


    # --- NEW: Export profiles for per-feature 25th & 75th percentile with others at mean ---
    @torch.no_grad()
    def export_profiles_feature_percentiles(mean_net,
                                            Xn, r_tags, device,
                                            key_y, basis_meta,
                                            out_dir="/home/gjc7gx/CovNNet/PercentileProfiles",
                                            fname_prefix="pred_profile"):
        """
        For each input dimension j:
            - Build x = mean(Xn, axis=0)
            - Set x[j] to the 25th percentile of Xn[:, j] → export profile
            - Set x[j] to the 75th percentile of Xn[:, j] → export profile
        Saves text files with columns: [radius_kpc, density_pred, v_c_pred]
        """
        os.makedirs(out_dir, exist_ok=True)

        # Precompute reference stats in normalized space
        x_mean = Xn.mean(axis=0).astype(np.float32)
        p25 = np.percentile(Xn, 25, axis=0).astype(np.float32)
        p75 = np.percentile(Xn, 75, axis=0).astype(np.float32)

        dx = Xn.shape[1]
        for j in range(dx):
            # 25th percentile for feature j
            x_p25 = x_mean.copy()
            x_p25[j] = p25[j]
            mu1_25, mu2_25 = _predict_bins_from_xn(mean_net, x_p25, device, key_y, basis_meta)
            out25 = np.column_stack([r_tags, mu1_25, mu2_25])
            fp25 = os.path.join(out_dir, f"{fname_prefix}_x{j:02d}_p25.txt")
            np.savetxt(fp25, out25, header="radius_kpc  density_pred  v_c_pred")

            # 75th percentile for feature j
            x_p75 = x_mean.copy()
            x_p75[j] = p75[j]
            mu1_75, mu2_75 = _predict_bins_from_xn(mean_net, x_p75, device, key_y, basis_meta)
            out75 = np.column_stack([r_tags, mu1_75, mu2_75])
            fp75 = os.path.join(out_dir, f"{fname_prefix}_x{j:02d}_p75.txt")
            np.savetxt(fp75, out75, header="radius_kpc  density_pred  v_c_pred")

        print(f"[export] Wrote per-feature p25/p75 profiles to {out_dir}")

    # Mean-input export (as before)
    export_profile_mean_input(
        mean_net, cov_net,
        Xn,                  # normalized dataset features
        r_tags, device,
        key_x=Xn_key, key_y=Yn_key,
        basis_meta=basis_meta,
        savepath="/home/gjc7gx/CovNNet/predicted_profile_mean_input.txt"
    )

    # NEW: Per-feature 25th/75th percentile exports with others held at mean
    export_profiles_feature_percentiles(
        mean_net,
        Xn, r_tags, device,
        key_y=Yn_key, basis_meta=basis_meta,
        out_dir="/home/gjc7gx/CovNNet/PercentileProfiles",
        fname_prefix="pred_profile"
    )

    # ------------------------------------------------------------------------------
    # Theoretical-model inversion from NN mean/cov  (DROP-IN REPLACEMENT)
    # ------------------------------------------------------------------------------
    from scipy.optimize import least_squares
    from scipy.integrate import cumulative_trapezoid
    import matplotlib.pyplot as plt

    # ----------------------------- Stable math utils ------------------------------
    def _clip_log(x, lo=-300.0, hi=300.0):
        """Clip values before np.exp to avoid overflow/underflow."""
        return np.clip(x, lo, hi)

    def _softplus_stable(x):
        """Numerically stable softplus."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        mask = x > 20.0
        out[mask] = x[mask]
        out[~mask] = np.log1p(np.exp(x[~mask]))
        return out

    def _soft_clip(x, lo, hi, softness=0.15):
        """
        Smoothly map R -> (lo,hi). softness is fraction of half-range.
        Prevents samples piling up hard on bounds (reduces 'stripes').
        """
        m  = 0.5 * (hi + lo)
        hw = 0.5 * (hi - lo)
        return m + hw * np.tanh((x - m) / (softness * hw + 1e-12))

    def sanitize_theta(theta, r, mode="fit"):
        """
        Enforce safe/physical parameter ranges.
        theta = [rho0, rs, gamma, beta, A, mu, sigma]
        mode="fit": minimal hard floors/clamps (friendly to optimizer)
        mode="sample": softplus + soft bounds (keeps variability finite)
        """
        theta = np.array(theta, dtype=np.float64, copy=True)
        rho0, rs, gamma, beta, A, mu, sigma = theta

        if mode == "fit":
            rho0  = max(rho0, 1e-12)
            rs    = max(rs,    1e-3)
            sigma = max(sigma, 1e-3)
            gamma = float(np.clip(gamma, 0.0, 2.0))
            beta  = float(np.clip(beta,  2.0, 6.0))
            A     = float(np.clip(A, -1e6, 1e6))
            log_rmin = float(np.log(np.clip(np.min(r), 1e-10, None)))
            log_rmax = float(np.log(np.max(r)))
            mu    = float(np.clip(mu, log_rmin - 5.0, log_rmax + 5.0))
        else:  # "sample"
            rho0  = _softplus_stable(rho0) + 1e-12
            rs    = _softplus_stable(rs)   + 1e-3
            sigma = _softplus_stable(sigma)+ 1e-3
            gamma = _soft_clip(gamma, 0.0, 2.0)
            beta  = _soft_clip(beta,  2.0, 6.0)
            A     = _soft_clip(A, -5e5, 5e5)
            log_rmin = float(np.log(np.clip(np.min(r), 1e-10, None)))
            log_rmax = float(np.log(np.max(r)))
            mu    = _soft_clip(mu, log_rmin - 3.0, log_rmax + 3.0)

        return np.array([rho0, rs, gamma, beta, A, mu, sigma], dtype=np.float64)

    # --------------------- Stable gNFW, M_enc, and v_rotation ---------------------
    def rho_gNFW_stable(r, rho0, rs, gamma, beta):
        """
        log ρ = log ρ0  - γ log x  + (γ-β) log(1+x),  x = r/rs
        Evaluate in log-space with guards.
        """
        r = np.asarray(r, dtype=np.float64)
        rs = max(rs, 1e-12)
        x = r / rs
        logx   = np.log(np.clip(x, 1e-300, None))
        log1px = np.log1p(x)
        log_rho = np.log(max(rho0, 1e-300)) - gamma * logx + (gamma - beta) * log1px
        log_rho = _clip_log(log_rho)
        rho = np.exp(log_rho)
        return np.nan_to_num(rho, nan=0.0, posinf=1e300, neginf=0.0)

    def enclosed_mass_from_rho_stable(r, rho):
        """
        M_enc(r) = 4π ∫ ρ(r') r'^2 dr'  (cumulative trapezoid).
        """
        r = np.asarray(r, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        r_safe = np.clip(r, 1e-10, None)
        rho = np.clip(np.nan_to_num(rho, nan=0.0, posinf=1e300), 0.0, 1e300)
        integrand = np.clip(4.0 * np.pi * rho * (r_safe**2), 0.0, 1e300)
        M_partial = cumulative_trapezoid(integrand, r_safe, initial=0.0)
        return np.clip(np.nan_to_num(M_partial, nan=0.0, posinf=1e300), 0.0, 1e300)

    def v_rotation_model_stable(r, rho0, rs, gamma, beta, A, mu, sigma, G=4.30091e-6):
        """
        v(r) = sqrt(G M_enc / r) + A / (r σ sqrt(2π)) * exp(-0.5 * ((log r - μ)/σ)^2).
        Units: r[kpc], ρ[Msun/kpc^3], G≈4.30091e-6 kpc km^2 s^-2 Msun^-1 → v[km/s]
        """
        r = np.asarray(r, dtype=np.float64)
        r_safe = np.clip(r, 1e-10, None)
        rho = rho_gNFW_stable(r_safe, rho0, rs, gamma, beta)
        M_enc = enclosed_mass_from_rho_stable(r_safe, rho)
        v_grav = np.sqrt(np.clip(G * M_enc / r_safe, 0.0, None))
        log_r = np.log(r_safe)
        sigma = max(sigma, 1e-6)
        bump = A / (r_safe * sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((log_r - mu) / sigma) ** 2)
        v = v_grav + bump
        return np.nan_to_num(v, nan=0.0, posinf=1e300)

    def stacked_theory_profiles(theta, r, mode="fit"):
        """
        Stable stacked profiles: y = [ρ(r), v(r)] with sanitized parameters.
        theta = [rho0, rs, gamma, beta, A, mu, sigma]
        """
        theta = sanitize_theta(theta, r, mode=mode)
        rho0, rs, gamma, beta, A, mu, sigma = theta
        y1 = rho_gNFW_stable(r, rho0, rs, gamma, beta)
        y2 = v_rotation_model_stable(r, rho0, rs, gamma, beta, A, mu, sigma)
        return np.concatenate([y1, y2], axis=0)

    # --------------- NN → (mean, cov) in physical bin space (Y1||Y2) --------------
    @torch.no_grad()
    def _predict_bins_and_cov_from_xn(mean_net, cov_net, xn_vec, device, key_y, basis_meta, Y1_len=50):
        """
        Returns (mu_bins [2M], Sigma_bins [2M,2M]) in physical bin space using your basis path.
        """
        xb = torch.from_numpy(xn_vec.astype(np.float32)).to(device).unsqueeze(0)
        mean_net.eval(); cov_net.eval()
        mu_n = mean_net(xb)                  # [1, d]
        L_n  = cov_net(xb)                   # [1, d, d]
        Sigma_n = L_n @ L_n.transpose(1, 2)  # [1, d, d]
        mu_n = mu_n.squeeze(0).cpu().numpy()
        Sigma_n = Sigma_n.squeeze(0).cpu().numpy()

        mu_coeff = _denorm(mu_n, key_y)                  # [d]
        scales_c = _linear_scales_from_key(key_y).astype(np.float64)
        D = np.diag(scales_c)
        Sigma_c = D @ Sigma_n @ D.T

        k1 = int(basis_meta["y1"]["k"])
        Phi1 = np.asarray(basis_meta["y1"]["Phi"], dtype=np.float64)
        Phi2 = np.asarray(basis_meta["y2"]["Phi"], dtype=np.float64)
        LOG1 = bool(basis_meta["y1"].get("logspace", False))
        LOG2 = bool(basis_meta["y2"].get("logspace", False))

        c1_mu, c2_mu = mu_coeff[:k1], mu_coeff[k1:]
        lin_mu1 = Phi1 @ c1_mu;  lin_mu2 = Phi2 @ c2_mu
        mu1 = np.exp(lin_mu1) if LOG1 else lin_mu1
        mu2 = np.exp(lin_mu2) if LOG2 else lin_mu2

        # Jacobians: coeffs -> bins
        J1 = (mu1[:, None]) * Phi1 if LOG1 else Phi1
        J2 = (mu2[:, None]) * Phi2 if LOG2 else Phi2
        Sigma_c11 = Sigma_c[:k1, :k1]
        Sigma_c22 = Sigma_c[k1:, k1:]
        Sigma_c12 = Sigma_c[:k1, k1:]
        S11 = J1 @ Sigma_c11 @ J1.T
        S22 = J2 @ Sigma_c22 @ J2.T
        S12 = J1 @ Sigma_c12 @ J2.T
        Sigma_bins = np.block([[S11, S12], [S12.T, S22]])  # [2M,2M]

        mu_bins = np.concatenate([mu1, mu2], axis=0)
        return mu_bins, Sigma_bins

    # ----------------------------- Fit θ̂ to NN mean ------------------------------
    def fit_theta_to_mu(mu_bins, Sigma_bins, r, theta_init, bounds=None):
        """
        Whitened least-squares with robust loss: minimize || W^(1/2)(y(theta) - mu_bins) ||_2.
        """
        ydim = mu_bins.size
        jitter = 1e-8 * np.eye(ydim)
        try:
            L = np.linalg.cholesky(Sigma_bins + jitter)
            Winv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            std = np.sqrt(np.clip(np.diag(Sigma_bins), 1e-18, None))
            Winv = np.diag(1.0 / std)

        def resid(theta):
            y_model = stacked_theory_profiles(theta, r, mode="fit")
            return Winv @ (np.clip(y_model, -1e12, 1e12) - np.clip(mu_bins, -1e12, 1e12))

        if bounds is not None:
            lb, ub = bounds
            lb = lb.copy(); ub = ub.copy()
            lb[:3] = np.maximum(lb[:3], [1e-12, 1e-3, 0.0])  # rho0, rs, gamma floors
            bounds = (lb, ub)

        res = least_squares(
            resid, x0=np.asarray(theta_init, dtype=np.float64),
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            method="trf", loss="soft_l1", f_scale=1.0,
            max_nfev=30000, ftol=1e-9, xtol=1e-9, gtol=1e-9
        )
        return res.x

    # -------------------- Finite-difference Jacobian: J=dy/dθ ---------------------
    def finite_diff_jacobian(theta_hat, r, eps_rel=None):
        theta_hat = np.asarray(theta_hat, dtype=np.float64)
        d = theta_hat.size
        y0 = stacked_theory_profiles(theta_hat, r, mode="fit")
        ydim = y0.size
        J = np.zeros((ydim, d), dtype=np.float64)
        if eps_rel is None:
            eps_rel = np.array([1e-4, 1e-4, 5e-4, 5e-4, 1e-4, 1e-4, 1e-4], dtype=np.float64)
        eps_rel = np.broadcast_to(eps_rel, (d,))
        for j in range(d):
            h = eps_rel[j] * (abs(theta_hat[j]) + 1.0)
            t_plus = theta_hat.copy();  t_plus[j]  += h
            t_minus = theta_hat.copy(); t_minus[j] -= h
            y_plus = stacked_theory_profiles(t_plus, r, mode="fit")
            y_minus = stacked_theory_profiles(t_minus, r, mode="fit")
            J[:, j] = (y_plus - y_minus) / (2.0 * h)
        return J  # [2M, d]

    # -------- Solve Σθ from Σy ≈ J Σθ Jᵀ (Tikhonov + PSD projection) --------------
    def solve_sigma_theta(J, Sigma_y, ridge=1e-3):
        """
        Solve vec(Sigma_y) ≈ (J ⊗ J) vec(Sigma_theta) with Tikhonov ridge.
        """
        d = J.shape[1]
        K = np.kron(J, J)
        r_vec = Sigma_y.reshape(-1)
        KT = K.T
        A = KT @ K + ridge * np.eye(d * d)
        b = KT @ r_vec
        s_vec = np.linalg.solve(A, b)
        S = s_vec.reshape(d, d)
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 0.0, None)
        return (V * w) @ V.T

    def corr_from_cov(S):
        std = np.sqrt(np.clip(np.diag(S), 1e-18, None))
        Dinv = np.diag(1.0 / std)
        return Dinv @ S @ Dinv

    # ------------------------- End-to-end driver for one X ------------------------
    @torch.no_grad()
    def infer_theory_params_for_xn(mean_net, cov_net, xn_vec, r_tags, device, key_y, basis_meta,
                                theta_init=None, bounds=None, print_prefix="[theory] "):
        mu_bins, Sigma_bins = _predict_bins_and_cov_from_xn(
            mean_net, cov_net, xn_vec, device, key_y, basis_meta, Y1_len=len(r_tags)
        )
        r = np.asarray(r_tags, dtype=np.float64)

        if theta_init is None:
            rho0_guess = max(1.0, np.median(mu_bins[:len(r)]))
            rs_guess   = 5.0
            theta_init = np.array([rho0_guess, rs_guess, 1.0, 3.0, 50.0, np.log(np.median(r)), 0.5],
                                dtype=np.float64)

        if bounds is None:
            lb = np.array([1e-12, 1e-3, 0.0,  2.0,   -1e6,  np.log(r.min())-5.0,  1e-3], dtype=np.float64)
            ub = np.array([1e20,  1e3,  2.0,  6.0,    1e6,  np.log(r.max())+5.0,  5.0 ], dtype=np.float64)
            bounds = (lb, ub)

        theta_hat = fit_theta_to_mu(mu_bins, Sigma_bins, r, theta_init, bounds=bounds)
        J = finite_diff_jacobian(theta_hat, r)
        Sigma_theta = solve_sigma_theta(J, Sigma_bins, ridge=1e-3)
        Corr_theta  = corr_from_cov(Sigma_theta)

        print(print_prefix + "theta_hat = [rho0, rs, gamma, beta, A, mu, sigma]:")
        print(np.array2string(theta_hat, precision=5, suppress_small=False))
        print(print_prefix + "Corr(theta) (rows/cols in same order):")
        with np.printoptions(precision=3, suppress=True):
            print(Corr_theta)

        outdir = "/home/gjc7gx/CovNNet/TheoryFit"
        os.makedirs(outdir, exist_ok=True)
        np.savetxt(os.path.join(outdir, "theta_hat.txt"), theta_hat,
                header="rho0  rs  gamma  beta  A  mu  sigma")
        np.savetxt(os.path.join(outdir, "Sigma_theta.txt"), Sigma_theta)
        np.savetxt(os.path.join(outdir, "Corr_theta.txt"), Corr_theta)
        print(print_prefix + f"Saved results to {outdir}")
        return theta_hat, Sigma_theta, Corr_theta

    # ---- Example call on mean normalized X ----
    x_mean_norm = Xn.mean(axis=0).astype(np.float32)
    _ = infer_theory_params_for_xn(
        mean_net, cov_net,
        xn_vec=x_mean_norm,
        r_tags=r_tags,
        device=device,
        key_y=Yn_key,         # coeff (output) normalization key
        basis_meta=basis_meta # contains Phi/logspace flags
    )

    # --------------------- Plot: NN vs Model (means + corr) -----------------------
    def _split_profiles(vec, M): return vec[:M], vec[M:]

    def _corr_from_cov_safe(S):
        S = np.asarray(S, dtype=np.float64)
        diag = np.clip(np.diag(S), 1e-18, None)
        Dinv = np.diag(1.0 / np.sqrt(diag))
        R = Dinv @ S @ Dinv
        return 0.5 * (np.clip(R, -1.0, 1.0) + np.clip(R.T, -1.0, 1.0))

    def _block_tick_labels(r):
        M = len(r); idxs = np.linspace(0, M - 1, num=min(8, M), dtype=int)
        ticks = list(idxs) + list(M + idxs)
        labels = [f"ρ({r[i]:.2g}kpc)" for i in idxs] + [f"v({r[i]:.2g}kpc)" for i in idxs]
        return ticks, labels

    def plot_nn_vs_model_profiles_and_corr(r, mu_bins, Sigma_bins, theta_hat, Sigma_theta, J,
                                        save_prefix="theory_vs_nn"):
        M = len(r)
        y_model = stacked_theory_profiles(theta_hat, r, mode="fit")
        rho_nn, v_nn = _split_profiles(mu_bins, M)
        rho_th, v_th = _split_profiles(y_model, M)

        Corr_nn = _corr_from_cov_safe(Sigma_bins)
        Sigma_model = J @ Sigma_theta @ J.T
        Corr_model = _corr_from_cov_safe(Sigma_model)

        fig = plt.figure(figsize=(12, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(r, rho_nn, lw=2, label="NN mean ρ(r)")
        ax1.plot(r, rho_th, lw=2, ls="--", label="Model mean ρ(r)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("r [kpc]"); ax1.set_ylabel("ρ [Msun/kpc³]")
        ax1.set_title("Density profile: NN vs Model"); ax1.legend(loc="best", frameon=False)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(r, v_nn, lw=2, label="NN mean v(r)")
        ax2.plot(r, v_th, lw=2, ls="--", label="Model mean v(r)")
        ax2.set_xscale("log"); ax2.set_xlabel("r [kpc]"); ax2.set_ylabel("v [km/s]")
        ax2.set_title("Rotation curve: NN vs Model"); ax2.legend(loc="best", frameon=False)

        ax3 = fig.add_subplot(gs[1, 0])
        im1 = ax3.imshow(Corr_nn, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax3.set_title("Corr (NN outputs)")
        ticks, labels = _block_tick_labels(r)
        ax3.set_xticks(ticks); ax3.set_xticklabels(labels, rotation=45, ha="right")
        ax3.set_yticks(ticks); ax3.set_yticklabels(labels)
        ax3.axvline(M - 0.5, color="k", lw=0.8); ax3.axhline(M - 0.5, color="k", lw=0.8)

        ax4 = fig.add_subplot(gs[1, 1])
        im2 = ax4.imshow(Corr_model, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax4.set_title("Corr (Model: J Σθ Jᵀ)")
        ax4.set_xticks(ticks); ax4.set_xticklabels(labels, rotation=45, ha="right")
        ax4.set_yticks(ticks); ax4.set_yticklabels(labels)
        ax4.axvline(M - 0.5, color="k", lw=0.8); ax4.axhline(M - 0.5, color="k", lw=0.8)

        cbar = fig.colorbar(im2, ax=[ax3, ax4], location="right", shrink=0.9, pad=0.02)
        cbar.set_label("Correlation")

        plt.savefig(f"{save_prefix}.png", dpi=160)
        plt.close(fig)
        print(f"[theory] Saved comparison figure to {save_prefix}.png")

    # Call right after infer_theory_params_for_xn(...) returns
    theta_hat, Sigma_theta, Corr_theta = _
    J = finite_diff_jacobian(theta_hat, np.asarray(r_tags, dtype=np.float64))
    mu_bins, Sigma_bins = _predict_bins_and_cov_from_xn(
        mean_net, cov_net, x_mean_norm, device, Yn_key, basis_meta, Y1_len=len(r_tags)
    )
    plot_nn_vs_model_profiles_and_corr(
        r=np.asarray(r_tags, dtype=np.float64),
        mu_bins=mu_bins, Sigma_bins=Sigma_bins,
        theta_hat=theta_hat, Sigma_theta=Sigma_theta,
        J=J,
        save_prefix="/home/gjc7gx/CovNNet/TheoryFit/theory_vs_nn"
    )

    # ----------------------- PSD-safe sampling & correlations ---------------------
    def _nearest_psd_eig(S, floor=1e-12, ridge=0.0):
        S_sym = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S_sym)
        w_clipped = np.maximum(w, floor)
        if ridge > 0.0: w_clipped = w_clipped + ridge
        return (V * w_clipped) @ V.T, w_clipped, V

    def _matrix_sqrt_from_eig(w, V, tol=1e-14):
        keep = w > tol
        if not np.any(keep):
            d = len(w)
            return np.zeros((d, d), dtype=np.float64), 0
        Wsqrt = np.sqrt(w[keep]); Vkeep = V[:, keep]
        return Vkeep * Wsqrt, keep.sum()

    def corr_empirical_safe(Y, eps=1e-8):
        """
        Robust empirical correlation; zeros rows/cols whose std < eps.
        """
        Y = np.asarray(Y, dtype=np.float64)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        std = Yc.std(axis=0, ddof=1)
        C = (Yc.T @ Yc) / max(Y.shape[0] - 1, 1)
        denom = np.outer(std, std)
        denom[denom < eps**2] = np.inf
        R = C / denom
        mask_const = std < eps
        R[mask_const, :] = 0.0; R[:, mask_const] = 0.0
        np.fill_diagonal(R, 1.0)
        return R, mask_const

    def sample_model_profile_distribution(theta_hat, Sigma_theta, r, n_samples=5000, seed=42,
                                        floor=1e-12, ridge_schedule=(0.0, 1e-10, 1e-9, 1e-8)):
        """
        Sample θ ~ N(θ̂, Σθ) robustly even if Σθ is not PD; evaluate profiles;
        return empirical mean/cov/corr in y-space.
        """
        rng = np.random.default_rng(seed)

        S_psd, w, V = _nearest_psd_eig(Sigma_theta, floor=floor, ridge=0.0)
        if float(w.min()) <= 0.0:
            for rid in ridge_schedule:
                S_psd, w, V = _nearest_psd_eig(Sigma_theta, floor=floor, ridge=rid)
                if w.min() > 0.0: break

        L, rnk = _matrix_sqrt_from_eig(w, V, tol=1e-14)
        print(f"[theory] Sigma_theta min-eig(after PSD proj)={w.min():.3e} | eff-rank={rnk}/{len(w)}")

        y_samps = []
        for _ in range(n_samples):
            z = rng.standard_normal(rnk)
            theta_draw = theta_hat + L @ z
            theta_draw = sanitize_theta(theta_draw, r, mode="sample")
            y_samps.append(stacked_theory_profiles(theta_draw, r, mode="sample"))

        Y = np.vstack(y_samps)  # [n_samples, 2M]
        y_mean = Y.mean(axis=0)
        y_cov  = np.cov(Y, rowvar=False)
        y_corr, mask_const = corr_empirical_safe(Y)
        print(f"[theory] near-constant bins in samples: {mask_const.sum()}")
        return y_mean, y_cov, y_corr

    # Sampled correlations vs NN
    y_mean_samp, y_cov_samp, y_corr_samp = sample_model_profile_distribution(
        theta_hat, Sigma_theta, np.asarray(r_tags), n_samples=5000
    )

    def estimate_theta_X_corr(mean_net, cov_net, Xn, r_tags, device, key_y, basis_meta,
                            theta_init=None, bounds=None, n_generated=1000, seed=0):
        """
        Compute correlations where:
        - Corr(theta, X) is determined by *synthetic inputs* generated around mean(X),
            each passed through the NN and fit for theta.
        - Corr(theta, theta) and Corr(X, X) are computed from those same synthetic draws.

        Args:
        mean_net, cov_net, Xn, r_tags, device, key_y, basis_meta : standard args
        theta_init : optional initial guess for fitting
        bounds     : optional bounds for fit
        n_generated : number of synthetic X samples to generate
        seed       : RNG seed

        Returns:
        Corr_theta_X : [d, p]
        Corr_theta   : [d, d]
        Corr_X       : [p, p]
        theta_names, x_labels : lists
        """
        rng = np.random.default_rng(seed)
        r = np.asarray(r_tags, dtype=np.float64)
        p = Xn.shape[1]

        # --------------------------
        # Step 1: determine feature ranges
        # --------------------------
        x_mean = Xn.mean(axis=0)
        x_min, x_max = Xn.min(axis=0), Xn.max(axis=0)
        x_range = x_max - x_min
        # ±30% of total feature range
        perturb_scale = 0.3 * x_range

        # --------------------------
        # Step 2: generate random synthetic X samples
        # --------------------------
        Xgen = x_mean + rng.uniform(low=-1.0, high=1.0, size=(n_generated, p)) * perturb_scale
        Xgen = np.clip(Xgen, 0.0, 1.0).astype(np.float32)  # still normalized range

        # --------------------------
        # Step 3: fit θ for each generated X
        # --------------------------
        if bounds is None:
            lb = np.array([1e-12, 1e-3, 0.0,  2.0,   -1e6,  np.log(r.min())-5.0,  1e-3], dtype=np.float64)
            ub = np.array([1e20,  1e3,  2.0,  6.0,    1e6,  np.log(r.max())+5.0,  5.0 ], dtype=np.float64)
            bounds = (lb, ub)

        thetas = []
        theta_prev = None
        for i in tqdm(range(n_generated)):
            xn_vec = Xgen[i]
            mu_bins_i, Sigma_bins_i = _predict_bins_and_cov_from_xn(
                mean_net, cov_net, xn_vec, device, key_y, basis_meta, Y1_len=len(r_tags)
            )

            if theta_prev is None:
                rho0_guess = max(1.0, np.median(mu_bins_i[:len(r)]))
                rs_guess   = 5.0
                theta_init_i = np.array([rho0_guess, rs_guess, 1.0, 3.0, 50.0,
                                        np.log(np.median(r)), 0.5], dtype=np.float64)
            else:
                theta_init_i = theta_prev

            theta_hat_i = fit_theta_to_mu(mu_bins_i, Sigma_bins_i, r, theta_init_i, bounds=bounds)
            thetas.append(theta_hat_i)
            theta_prev = theta_hat_i.copy()

        Theta = np.vstack(thetas).astype(np.float64)
        Xused = Xgen.astype(np.float64)

        # --------------------------
        # Step 4: compute correlations
        # --------------------------
        Tz = (Theta - Theta.mean(axis=0)) / (Theta.std(axis=0, ddof=1) + 1e-12)
        Xz = (Xused - Xused.mean(axis=0)) / (Xused.std(axis=0, ddof=1) + 1e-12)

        n = Tz.shape[0]
        Corr_theta_X = (Tz.T @ Xz) / (n - 1)
        Corr_theta   = (Tz.T @ Tz) / (n - 1)
        Corr_X       = (Xz.T @ Xz) / (n - 1)
        np.fill_diagonal(Corr_theta, 1.0)
        np.fill_diagonal(Corr_X, 1.0)

        # --------------------------
        # Step 5: labeling
        # --------------------------
        theta_names = [r"$\rho_0$", r"$r_s$", r"$\gamma$", r"$\beta$", r"$A$", r"$\mu$", r"$\sigma$"]
        try:
            x_labels = list(X_cols)
            if len(x_labels) != p:
                raise ValueError
        except Exception:
            x_labels = [f"X{j}" for j in range(p)]

        print(f"[θ–X synthetic] Generated {n_generated} random X inputs ±30% range | "
            f"Fit {len(Theta)} θ samples successfully.")
        return Corr_theta_X, Corr_theta, Corr_X, theta_names, x_labels


    # --------------------------------------------------------------------------
    # Build combined θ–X correlation block with independent row/col order flips
    # --------------------------------------------------------------------------
    def _build_combined_corr_block(
        C_theta_X, C_theta, C_X, theta_names, x_labels,
        reverse_theta_rows=False,
        reverse_theta_cols=True,
        reverse_x_rows=True,
        reverse_x_cols=False
    ):
        """
        Construct one block correlation with rows=[θ; X], cols=[X; θ],
        but allow independent reversing of θ/X order on rows vs cols.

        Returns:
            C_full : combined matrix
            row_labels, col_labels
            row_split, col_split : split positions for quadrant lines
        """
        d = C_theta.shape[0]
        p = C_X.shape[0]

        # --- choose row/col orders independently
        theta_row_order = np.arange(d-1, -1, -1) if reverse_theta_rows else np.arange(d)
        theta_col_order = np.arange(d-1, -1, -1) if reverse_theta_cols else np.arange(d)
        x_row_order     = np.arange(p-1, -1, -1) if reverse_x_rows     else np.arange(p)
        x_col_order     = np.arange(p-1, -1, -1) if reverse_x_cols     else np.arange(p)

        # --- assemble blocks
        top_left  = C_theta_X[theta_row_order][:, x_col_order]    # θ–X
        top_right = C_theta[theta_row_order][:, theta_col_order]  # θ–θ
        bot_left  = C_X[x_row_order][:, x_col_order]              # X–X
        bot_right = C_theta_X.T[x_row_order][:, theta_col_order]  # X–θ

        C_full = np.block([
            [top_left,  top_right],
            [bot_left,  bot_right],
        ])

        row_labels = [theta_names[j] for j in theta_row_order] + [x_labels[i] for i in x_row_order]
        col_labels = [x_labels[i]   for i in x_col_order]      + [theta_names[j] for j in theta_col_order]

        row_split = len(theta_row_order) - 0.5
        col_split = len(x_col_order) - 0.5

        return C_full, row_labels, col_labels, row_split, col_split


    # --------------------------------------------------------------------------
    # Plotting function with combined bottom block
    # --------------------------------------------------------------------------
    def plot_nn_vs_model_profiles_and_corr(
        r,
        mu_bins, Sigma_bins,
        theta_hat,
        Corr_model,                  # sampled correlation of profiles [2M,2M]
        Corr_theta_X,                # [d, p]
        Corr_theta,                  # [d, d]
        Corr_X,                      # [p, p]
        theta_names, x_labels,
        corr_model_title="Corr (Model, sampled)",
        save_prefix="theory_vs_nn"
    ):
        M = len(r)
        y_model = stacked_theory_profiles(theta_hat, r, mode="fit")
        rho_nn, v_nn = mu_bins[:M], mu_bins[M:]
        rho_th, v_th = y_model[:M], y_model[M:]
        Corr_nn = _corr_from_cov_safe(Sigma_bins)

        fig = plt.figure(figsize=(14, 15), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.8])

        # ---------------- Row 1: profiles ----------------
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(r, rho_nn, lw=2, label="NN mean ρ(r)")
        ax1.plot(r, rho_th, lw=2, ls="--", label="Model mean ρ(r)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("r [kpc]"); ax1.set_ylabel("ρ [Msun/kpc³]")
        ax1.set_title("Density profile: NN vs Model"); ax1.legend(loc="best", frameon=False)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(r, v_nn, lw=2, label="NN mean v(r)")
        ax2.plot(r, v_th, lw=2, ls="--", label="Model mean v(r)")
        ax2.set_xscale("log"); ax2.set_xlabel("r [kpc]"); ax2.set_ylabel("v [km/s]")
        ax2.set_ylim(0,500)
        ax2.set_title("Rotation curve: NN vs Model"); ax2.legend(loc="best", frameon=False)

        # ---------------- Row 2: correlation heatmaps ----------------
        ticks, labels = _block_tick_labels(r)
        ax3 = fig.add_subplot(gs[1, 0])
        im1 = ax3.imshow(Corr_nn, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax3.set_title("Corr (NN)")
        ax3.set_xticks(ticks); ax3.set_xticklabels(labels, rotation=45, ha="right")
        ax3.set_yticks(ticks); ax3.set_yticklabels(labels)
        ax3.axvline(M - 0.5, color="k", lw=0.8); ax3.axhline(M - 0.5, color="k", lw=0.8)

        ax4 = fig.add_subplot(gs[1, 1])
        im2 = ax4.imshow(Corr_model, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax4.set_title(corr_model_title)
        ax4.set_xticks(ticks); ax4.set_xticklabels(labels, rotation=45, ha="right")
        ax4.set_yticks(ticks); ax4.set_yticklabels(labels)
        ax4.axvline(M - 0.5, color="k", lw=0.8); ax4.axhline(M - 0.5, color="k", lw=0.8)

        cbar2 = fig.colorbar(im2, ax=[ax3, ax4], location="right", shrink=0.9, pad=0.02)
        cbar2.set_label("Correlation")

        # ---------------- Row 3: combined θ–X correlation block ----------------
        C_full, row_labels, col_labels, rsplit, csplit = _build_combined_corr_block(
            Corr_theta_X, Corr_theta, Corr_X, theta_names, x_labels,
            reverse_theta_rows=False,
            reverse_theta_cols=True,
            reverse_x_rows=True,   # X0 at top for rows
            reverse_x_cols=False   # X0 at left for cols
        )


        ax5 = fig.add_subplot(gs[2, :])
        im3 = ax5.imshow(C_full, vmin=-1, vmax=1, origin="lower",
                        interpolation="nearest", aspect="auto")
        ax5.invert_yaxis()
        ax5.set_title("Combined correlations (rows = [θ; X], cols = [X; θ])")

        ax5.set_yticks(np.arange(len(row_labels)))
        ax5.set_yticklabels(row_labels)
        ax5.set_xticks(np.arange(len(col_labels)))
        ax5.set_xticklabels(col_labels, rotation=45, ha="right")

        ax5.axhline(rsplit, color="k", lw=0.8)
        ax5.axvline(csplit, color="k", lw=0.8)

        cbar3 = fig.colorbar(im3, ax=[ax5], location="right", shrink=0.9, pad=0.02)
        cbar3.set_label("Correlation")

        plt.savefig(f"{save_prefix}.png", dpi=160)
        plt.close(fig)
        print(f"[theory] Saved comparison figure to {save_prefix}.png")

    # ------------------------------------------------------------------
    # 1. Build dataset-level correlations (θ–X, θ–θ, X–X)
    # ------------------------------------------------------------------
    Corr_theta_X, Corr_theta_ds, Corr_X, theta_names, x_labels = estimate_theta_X_corr(
        mean_net, cov_net,
        Xn=Xn,                      # used only for mean/range reference
        r_tags=r_tags,
        device=device,
        key_y=Yn_key,
        basis_meta=basis_meta,
        theta_init=None,            # optional, can be left None
        bounds=None,                # optional, will auto-generate safe defaults
        n_generated=10000,            # number of random X samples (adjust as needed)
        seed=0
    )
    # ------------------------------------------------------------------
    # 2. Plot with combined bottom block
    # ------------------------------------------------------------------
    plot_nn_vs_model_profiles_and_corr(
        r=np.asarray(r_tags, dtype=np.float64),
        mu_bins=mu_bins,
        Sigma_bins=Sigma_bins,
        theta_hat=theta_hat,
        Corr_model=y_corr_samp,                       # sampled profile corr
        Corr_theta_X=Corr_theta_X,
        Corr_theta=Corr_theta_ds,
        Corr_X=Corr_X,
        theta_names=theta_names,
        x_labels=x_labels,
        corr_model_title="Corr (Model, sampled)",
        save_prefix="/home/gjc7gx/CovNNet/TheoryFit/theory_vs_nn"
    )

