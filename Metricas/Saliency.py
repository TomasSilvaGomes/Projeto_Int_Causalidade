"""
saliency_evaluation.py
Avalia o método XAI Saliency (gradiente simples) com as métricas:
Monotonicity, Sparseness e MaxSensitivity.
"""

import sys
import os

# =========================================================================
# 1. CORREÇÃO DE CAMINHOS: CALCULAR A RAIZ DO PROJETO DE FORMA ABSOLUTA
# =========================================================================

# Adiciona a raiz do projeto (Pasta superior ao script atual) ao sys.path
# Isto permite importar "Rede.rede_pytorch"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from Rede.rede_pytorch import CNN, train_and_save_model


# === CONFIGURAÇÃO GERAL ====================================================

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# Caminho para o dataset MNIST existente
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_data'))

# Caminho para guardar resultados
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Avaliacao_Metricas'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminho para o modelo treinado
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Rede', 'mnist_cnn_pytorch.pth'))


# === FUNÇÕES DE SUPORTE ====================================================

def load_model_and_data():
    """Carrega o modelo CNN treinado e 10 amostras do MNIST (uma por classe)."""
    model = train_and_save_model(model_path=MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    x_list, y_list = [], []
    for label in range(10):
        idx = (dataset.targets == label).nonzero(as_tuple=True)[0][0].item()
        x, y = dataset[idx]
        x_list.append(x)
        y_list.append(y)

    x_batch = torch.stack(x_list).to(device)
    y_batch = torch.tensor(y_list).to(device)

    return model, x_batch, y_batch, device


# === MÉTODO: SALIENCY =====================================================

def saliency(model, x, target):
    """
    Calcula o mapa de saliência (gradiente da saída em relação ao input).
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    output = model(x)
    selected = output[torch.arange(len(target)), target].sum()

    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    selected.backward(retain_graph=False)
    grad = x.grad.detach()
    return grad


# === MÉTRICAS ===============================================================

def continuity(model, x, target, attr, step=1):
    """
    Avalia a continuidade da explicação.
    Mede se a confiança do modelo diminui de forma consistente
    quando removemos características importantes do input.
    """
    with torch.no_grad():
        pred_orig = model(x).softmax(1)[torch.arange(len(target)), target]
    scores = []
    flat_attr = attr.flatten(2)
    for i in range(0, flat_attr.shape[2], step):
        mask = flat_attr.clone()
        mask[:, :, i:] = 0
        masked_x = x * mask.view_as(x)
        with torch.no_grad():
            pred_masked = model(masked_x).softmax(1)[torch.arange(len(target)), target]
        scores.append((pred_orig - pred_masked).mean().item())
    return np.mean(scores)


def selectivity(model, x, target, attr, step=10):
    """
    Mede a seletividade da explicação.
    Remove progressivamente regiões mais relevantes do input e mede
    a queda da confiança do modelo na classe correta.
    """
    with torch.no_grad():
        pred_orig = model(x).softmax(1)[torch.arange(len(target)), target]

    scores = []
    attr_flat = attr.flatten(1)

    for i in range(0, attr_flat.shape[1], step):
        # Cria máscara removendo as i características mais relevantes
        topk_indices = attr_flat.topk(i, dim=1).indices if i > 0 else torch.tensor([], device=attr.device)
        mask = torch.ones_like(attr_flat)
        if i > 0:
            mask.scatter_(1, topk_indices, 0)
        masked_x = (x.flatten(1) * mask).view_as(x)
        with torch.no_grad():
            pred_masked = model(masked_x).softmax(1)[torch.arange(len(target)), target]
        scores.append((pred_orig - pred_masked).mean().item())

    return np.mean(scores)


def ROAD(model, x, target, attr_func, n_samples=10, noise_std=0.3):
    """
    Mede a robustez da explicação face a pequenas perturbações no input.
    Corresponde ao conceito de ROAD: Remove And Debias.
    """
    base_attr = attr_func(model, x, target)
    max_diff = 0
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        noisy_x = x + noise
        noisy_attr = attr_func(model, noisy_x, target)
        diff = torch.abs(base_attr - noisy_attr).mean().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff


# === AVALIAÇÃO FINAL =======================================================

def evaluate_saliency():
    print("\nIniciando avaliação com Saliency...")
    model, x_batch, y_batch, device = load_model_and_data()

    attr = saliency(model, x_batch, y_batch)

    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, saliency)
    }

    df = pd.DataFrame([results], index=["Saliency"])
    csv_path = os.path.join(OUTPUT_DIR, "Saliency_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)
    print(f"\nSalvo em: {csv_path}")
    return df


if __name__ == "__main__":
    evaluate_saliency()
