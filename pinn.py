# pinn.py
"""
Pure PINN (Physics-Informed Neural Network) solver for the same 1D viscous
Burgers' problem, as an independent (mesh-free, non-quantum) comparison point.

No Cole-Hopf trick, no time-stepping grid: a network u_theta(x, t) is trained
so that its automatic-differentiation residual of the PDE, plus the IC and BC
mismatch, all go to zero. Evaluate u_theta at t = T_FINAL and compare to the
classical FTCS reference.

Requires: pip install torch --break-system-packages
"""

import numpy as np
import torch
import torch.nn as nn
from params import *   # x, dx, dt, N, N_STEPS, NU, u_init

torch.manual_seed(0)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

X_MIN, X_MAX = float(x[0]), float(x[-1])
T_FINAL = N_STEPS * dt


class PINN(nn.Module):
    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers += [nn.Linear(n_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def pde_residual(model, x_f, t_f, nu):
    x_f = x_f.clone().requires_grad_(True)
    t_f = t_f.clone().requires_grad_(True)
    u = model(x_f, t_f)
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def train_pinn(n_epochs=5000, n_f=2000, lr=1e-3, verbose=True):
    model = PINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # interior collocation points, random in space-time
    x_f = (torch.rand(n_f, 1) * (X_MAX - X_MIN) + X_MIN).to(device)
    t_f = (torch.rand(n_f, 1) * T_FINAL).to(device)

    # initial condition: t = 0, x = grid
    x_ic = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic).to(device)
    u_ic = torch.tensor(u_init(x), dtype=torch.float32).view(-1, 1).to(device)

    # boundary: x = X_MIN and x = X_MAX, random t
    t_bc = (torch.rand(200, 1) * T_FINAL).to(device)
    x_bc_l = torch.full_like(t_bc, X_MIN).to(device)
    x_bc_r = torch.full_like(t_bc, X_MAX).to(device)

    for epoch in range(n_epochs):
        opt.zero_grad()

        loss_pde = torch.mean(pde_residual(model, x_f, t_f, NU) ** 2)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_ic) ** 2)
        loss_bc = torch.mean(model(x_bc_l, t_bc) ** 2) + torch.mean(model(x_bc_r, t_bc) ** 2)

        loss = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
        loss.backward()
        opt.step()

        if verbose and epoch % 500 == 0:
            print(f"epoch {epoch:5d}  loss {loss.item():.3e}  "
                  f"(pde {loss_pde.item():.2e}, ic {loss_ic.item():.2e}, bc {loss_bc.item():.2e})")

    return model


def evaluate_pinn(model, x_grid, t_final):
    model.eval()
    x_t = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)
    t_t = torch.full_like(x_t, t_final)
    with torch.no_grad():
        u = model(x_t, t_t).cpu().numpy().flatten()
    return u


if __name__ == "__main__":
    from classical import u_hist

    model = train_pinn()
    u_pinn_final = evaluate_pinn(model, x, T_FINAL)

    l2 = np.sqrt(np.mean((u_hist[-1] - u_pinn_final) ** 2))
    print(f"L2 error (PINN vs classical): {l2:.2e}")
