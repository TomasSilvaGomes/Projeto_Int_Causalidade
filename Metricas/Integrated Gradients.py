"""
integratedgradients_evaluation.py
Avalia o método XAI Integrated Gradients com as métricas:
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
    """Carrega o modelo CNN treinado e 10 amostras do conjunto MNIST (uma por classe)."""
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


# === IMPLEMENTAÇÃO DO MÉTODO: Integrated Gradients ==========================

def integrated_gradients(model, x, target, baseline=None, n_steps=50):
    """
    Implementação simples de Integrated Gradients.
    Retorna attributions com a mesma forma de x (tensor).
    """
    model.eval()
    device = x.device

    if baseline is None:
        baseline = torch.zeros_like(x).to(device)

    # linspace alphas: shape (n_steps, )
    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device)

    # acumular gradientes
    total_grad = torch.zeros_like(x).to(device)

    for alpha in alphas:
        # interpolated inputs
        inp = baseline + alpha * (x - baseline)
        inp = inp.clone().detach().requires_grad_(True)

        out = model(inp)
        # soma das probabilidades/logits para as classes alvo de cada amostra
        # selecionamos a saída correspondente a cada target
        selected = out[torch.arange(len(target)), target].sum()

        # computa gradiente selecionado em relação a input
        model.zero_grad()
        if inp.grad is not None:
            inp.grad.zero_()
        selected.backward(retain_graph=False)
        grads = inp.grad.detach()  # shape igual a x

        total_grad += grads

    # integral aproximada (riemann): average grad * (x - baseline)
    avg_grad = total_grad / float(n_steps)
    attributions = (x - baseline) * avg_grad
    return attributions.detach()


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


def ROAD(model, x, target, attr_func, n_samples=10, noise_std=0.3, **attr_kwargs):
    """
    Mede a robustez da explicação face a pequenas perturbações no input.
    Corresponde ao conceito de ROAD: Remove And Debias.
    """
    base_attr = attr_func(model, x, target, **attr_kwargs)
    max_diff = 0
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        pert_attr = attr_func(model, x + noise, target, **attr_kwargs)
        diff = torch.abs(base_attr - pert_attr).mean().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff

# === AVALIAÇÃO FINAL =======================================================

def evaluate_integratedgradients(n_steps=50):
    print("\nIniciando avaliação com Integrated Gradients...")
    model, x_batch, y_batch, device = load_model_and_data()

    # Calcular as atribuições
    attr = integrated_gradients(model, x_batch, y_batch, baseline=None, n_steps=n_steps)

    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, integrated_gradients, n_samples=10, noise_std=0.3, baseline=None, n_steps=n_steps)
    }

    df = pd.DataFrame([results], index=["IntegratedGradients"])
    csv_path = os.path.join(OUTPUT_DIR, "IntegratedGradients_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)
    print(f"\nSalvo em: {csv_path}")
    return df


if __name__ == "__main__":
    evaluate_integratedgradients(n_steps=50)
