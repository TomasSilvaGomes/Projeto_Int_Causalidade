"""
integratedgradients_evaluation.py
Avalia o método XAI Integrated Gradients com as métricas:
Monotonicity, Sparseness e MaxSensitivity.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from Rede.rede_pytorch import CNN, train_and_save_model
from typing import Callable
from DeepLift import *

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


def integrated_gradients(model, x, target, baseline=None, n_steps=50, **kwargs):
    """
    Implementação manual de Integrated Gradients.
    Aceita **kwargs para compatibilidade com a chamada genérica do ROAD.
    """
    model.eval()
    device = x.device

    if baseline is None:
        baseline = torch.zeros_like(x).to(device)

    # Criação dos alphas (n_steps)
    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device)

    # Acumulador de gradientes
    total_grad = torch.zeros_like(x).to(device)

    for alpha in alphas:
        # Input interpolado
        inp = baseline + alpha * (x - baseline)
        inp = inp.clone().detach().requires_grad_(True)

        out = model(inp)
        
        # Seleciona o score da classe alvo
        selected = out[torch.arange(len(target)), target].sum()

        model.zero_grad()
        if inp.grad is not None:
            inp.grad.zero_()
        selected.backward(retain_graph=False)
        
        grads = inp.grad.detach()
        total_grad += grads

    # Integral aproximada (Riemann): Média dos gradientes * (Input - Baseline)
    avg_grad = total_grad / float(n_steps)
    attributions = (x - baseline) * avg_grad
    
    return attributions.detach()


# === AVALIAÇÃO E VISUALIZAÇÃO POR LINHAS =====================================

def evaluate_integratedgradients(n_steps=50):
    print("\nIniciando avaliação com Integrated Gradients...")
    model, x_batch, y_batch, device = load_model_and_data()

    # Cálculo inicial da atribuição
    attr = integrated_gradients(model, x_batch, y_batch, baseline=None, n_steps=n_steps)

    # 1. CÁLCULO DAS MÉTRICAS
    # ROAD (Fidelidade): Passamos 'n_steps' do IG via kwargs
    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, integrated_gradients, 
                     n_steps_metric=50, 
                     percent_to_remove=0.25,
                     n_steps=n_steps, baseline=None) # Args específicos do IG
    }

    df = pd.DataFrame([results], index=["Integrated Gradients"])
    csv_path = os.path.join(OUTPUT_DIR, "IntegratedGradients_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)

    # === VISUALIZAÇÃO (3 LINHAS x 4 COLUNAS) =================================
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    label_to_show = 0
    fig.suptitle(f"Análise Visual das Métricas: Integrated Gradients (Label {label_to_show})", fontsize=22, fontweight='bold', y=0.98)

    sample_idx = label_to_show 
    x_sample = x_batch[sample_idx].unsqueeze(0)
    y_sample = y_batch[sample_idx].unsqueeze(0)
    attr_sample = attr[sample_idx].unsqueeze(0)

    def denormalize(img):
        img_np = img.cpu().numpy().squeeze()
        return np.clip((img_np * MNIST_STD + MNIST_MEAN), 0, 1)

    # --- LINHA 1: SELECTIVITY (Fidelidade Pontual) ---
    
    row = 0
    # Col 1: Original
    axes[row, 0].imshow(denormalize(x_sample), cmap='gray')
    axes[row, 0].set_ylabel("SELECTIVITY\n(Fidelidade)", fontsize=16, fontweight='bold', labelpad=20)
    axes[row, 0].set_title("Input Original", fontsize=12)
    
    # Col 2: Saliência
    attr_vis = attr_sample.squeeze().abs().cpu().numpy()
    axes[row, 1].imshow(denormalize(x_sample), cmap='gray', alpha=0.5)
    axes[row, 1].imshow(attr_vis, cmap='jet', alpha=0.6)
    axes[row, 1].set_title("Foco de Atribuição", fontsize=12)
    
    # Col 3: Máscara (Top 20%)
    attr_flat = attr_sample.flatten(1).abs()
    topk = int(0.20 * attr_flat.shape[1])
    topk_indices = attr_flat.topk(topk, dim=1).indices
    mask_sel = torch.ones_like(attr_flat)
    mask_sel.scatter_(1, topk_indices, 0) 
    
    mask_display = torch.zeros_like(attr_flat)
    mask_display.scatter_(1, topk_indices, 1) 
    axes[row, 2].imshow(mask_display.view_as(x_sample).squeeze().cpu().numpy(), cmap='Reds')
    axes[row, 2].set_title("Áreas Removidas (Top 20%)", fontsize=12)
    
    # Col 4: Resultado + Score
    masked_x_sel = (x_sample.flatten(1) * mask_sel).view_as(x_sample)
    axes[row, 3].imshow(denormalize(masked_x_sel), cmap='gray')
    axes[row, 3].set_title(f"Score Calculado: {results['Selectivity']:.4f}", fontsize=14, fontweight='bold', color='green')
    axes[row, 3].set_xlabel("Maior queda de confiança\nimplica melhor Selectivity.", fontsize=11)


    # --- LINHA 2: CONTINUITY (Estabilidade/Robustez) ---
    
    row = 1
    # Col 1: Original
    axes[row, 0].imshow(denormalize(x_sample), cmap='gray')
    axes[row, 0].set_ylabel("CONTINUITY\n(Estabilidade)", fontsize=16, fontweight='bold', labelpad=20)
    axes[row, 0].set_title("Input Original", fontsize=12)
    
    # Col 2: Ruído
    noise = torch.randn_like(x_sample) * 0.5
    x_noisy = x_sample + noise
    axes[row, 1].imshow(denormalize(x_noisy), cmap='gray')
    axes[row, 1].set_title("Input com Ruído", fontsize=12)
    
    # Col 3: Saliência no Ruído
    # Recalcula atribuição com os mesmos parâmetros (n_steps)
    attr_noisy = integrated_gradients(model, x_noisy, y_sample, baseline=None, n_steps=n_steps)
    attr_noisy_vis = attr_noisy.squeeze().abs().cpu().numpy()
    axes[row, 2].imshow(denormalize(x_noisy), cmap='gray', alpha=0.5)
    axes[row, 2].imshow(attr_noisy_vis, cmap='jet', alpha=0.6)
    axes[row, 2].set_title("Saliência sob Ruído", fontsize=12)
    
    # Col 4: Diferença + Score
    diff = torch.abs(attr_sample - attr_noisy).squeeze().cpu().numpy()
    im_diff = axes[row, 3].imshow(diff, cmap='magma')
    axes[row, 3].set_title(f"Score Calculado: {results['Continuity']:.4f}", fontsize=14, fontweight='bold', color='blue')
    axes[row, 3].set_xlabel("Menor diferença visual\nimplica melhor Continuity.", fontsize=11)


    # --- LINHA 3: ROAD (Fidelidade Cumulativa/AOPC) ---
    
    row = 2
    # Col 1: Original
    axes[row, 0].imshow(denormalize(x_sample), cmap='gray')
    axes[row, 0].set_ylabel("ROAD (AOPC)\n(Fidelidade)", fontsize=16, fontweight='bold', labelpad=20)
    axes[row, 0].set_title("Input Original", fontsize=12)
    
    # Col 2: Saliência 
    axes[row, 1].imshow(denormalize(x_sample), cmap='gray', alpha=0.5)
    axes[row, 1].imshow(attr_vis, cmap='jet', alpha=0.6)
    axes[row, 1].set_title("Base da Decisão", fontsize=12)
    
    # Col 3: Máscara Agressiva
    topk_road = int(0.50 * attr_flat.shape[1])
    topk_idx_road = attr_flat.topk(topk_road, dim=1).indices
    mask_road = torch.ones_like(attr_flat)
    mask_road.scatter_(1, topk_idx_road, 0)
    
    mask_vis_road = torch.zeros_like(attr_flat)
    mask_vis_road.scatter_(1, topk_idx_road, 1)
    axes[row, 2].imshow(mask_vis_road.view_as(x_sample).squeeze().cpu().numpy(), cmap='Reds')
    axes[row, 2].set_title("Remoção Cumulativa (50%)", fontsize=12)
    
    # Col 4: Resultado + Score
    masked_x_road = (x_sample.flatten(1) * mask_road).view_as(x_sample)
    axes[row, 3].imshow(denormalize(masked_x_road), cmap='gray')
    axes[row, 3].set_title(f"Score Calculado: {results['ROAD']:.4f}", fontsize=14, fontweight='bold', color='red')
    axes[row, 3].set_xlabel("Média da queda de confiança\nao longo da curva de remoção.", fontsize=11)


    # Ajustes finais
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"IntegratedGradients_Avaliacao_Label{label_to_show}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualização salva em: {plot_path}")
    plt.close()
    
    return df

if __name__ == "__main__":
    evaluate_integratedgradients()