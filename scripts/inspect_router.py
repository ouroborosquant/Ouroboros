import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.alpha.gat_signal_router import SignalRouterGAT, N_SIGNALS, SIGNAL_NAMES

# Cargar el modelo
device = "cpu"
router = SignalRouterGAT(n_signals=N_SIGNALS).to(device)
ckpt = torch.load("models/weights/gat_router.pt", map_location=device)
router.load_state_dict(ckpt["model_state_dict"])
router.eval()

# Simulamos una entrada (pesos de un día cualquiera)
# ... esto nos dirá si el router tiene sesgos o si está "plano"
print("Router weights initialized. Inspecting layer parameters...")
for name, param in router.named_parameters():
    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")