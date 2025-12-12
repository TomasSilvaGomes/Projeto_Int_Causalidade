"""
inputxgradient_evaluation.py
Avalia o método XAI Input×Gradient com as métricas:
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
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from Rede.rede_pytorch import CNN, train_and_save_model
from typing import Callable
from DeepLift import *


# === CONFIGURAÇÃO ==========================================================

# Constantes de Normalização do MNIST
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# =========================================================================
# 2. CORREÇÃO DE CONSTANTES: USAR CAMINHOS ABSOLUTOS BASEADOS NA RAIZ
# =========================================================================

# Configuração da Pasta
# OUTPUT_DIR será 'Projeto_Int_Causalidade/Imagens'
# Caminho para o dataset MNIST existente

# Caminho para o dataset MNIST existente
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sample_data'))

# Caminho para guardar resultados
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Avaliacao_Metricas'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminho para o modelo treinado
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Rede', 'mnist_cnn_pytorch.pth'))



def input_x_gradient(model, x, target, **kwargs):
    """
    Implementação do Input x Gradient.
    Atribuição = Input * Gradiente(Output/Input).
    Aceita **kwargs para compatibilidade com chamadas genéricas, mas não os usa.
    """
    # Clone e detach para garantir que não afetamos o grafo original
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = model(x)
    
    # Selecionar o score da classe alvo
    score = output[torch.arange(len(target)), target].sum()
    
    # Zerar gradientes anteriores e calcular o novo
    model.zero_grad()
    score.backward()
    
    # Input * Gradiente
    attribution = x * x.grad
    
    return attribution.detach()


# === AVALIAÇÃO E VISUALIZAÇÃO POR LINHAS =====================================

def evaluate_inputxgradient():
    print("\nIniciando avaliação com Input x Gradient...")
    model, x_batch, y_batch, device = load_model_and_data()

    # Cálculo inicial da atribuição para uso nas métricas simples
    attr = input_x_gradient(model, x_batch, y_batch)

    # 1. CÁLCULO DAS MÉTRICAS
    # ROAD agora usa os argumentos de Fidelidade (percent_to_remove)
    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, input_x_gradient, 
                     n_steps_metric=50, 
                     percent_to_remove=0.25) # InputXGrad não requer args extras como 'layer'
    }

    df = pd.DataFrame([results], index=["Input x Gradient"])
    csv_path = os.path.join(OUTPUT_DIR, "InputXGradient_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)

    # === VISUALIZAÇÃO (3 LINHAS x 4 COLUNAS) =================================
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    label_to_show = 0
    fig.suptitle(f"Análise Visual das Métricas: Input x Gradient (Label {label_to_show})", fontsize=22, fontweight='bold', y=0.98)

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
    attr_noisy = input_x_gradient(model, x_noisy, y_sample)
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
    plot_path = os.path.join(OUTPUT_DIR, f"InputXGradient_Avaliacao_Label{label_to_show}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualização salva em: {plot_path}")
    plt.close()
    
    return df

if __name__ == "__main__":
    evaluate_inputxgradient()
