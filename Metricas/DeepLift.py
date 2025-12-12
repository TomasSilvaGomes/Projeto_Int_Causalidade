"""
deeplift_evaluation.py
Avalia o método XAI DeepLIFT com as métricas:
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
from quantus import Selectivity
from torchvision import datasets, transforms
from Rede.rede_pytorch import CNN, train_and_save_model
from typing import Callable
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# === IMPLEMENTAÇÃO DO MÉTODO: DeepLIFT =====================================

def deeplift(model, x, target, baseline=None):
    """
    Implementação simplificada do DeepLIFT.
    Compara a ativação de cada neurónio com a de uma baseline (zero ou média).
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    # baseline (referência) — por padrão, uma imagem de zeros
    if baseline is None:
        baseline = torch.zeros_like(x).to(x.device)

    # forward original e baseline
    output_x = model(x)
    output_b = model(baseline)

    # diferença de outputs
    delta_out = (output_x - output_b)[torch.arange(len(target)), target]

    # gradiente normal
    grads = torch.autograd.grad(delta_out.sum(), x, retain_graph=False)[0]

    # diferença de inputs
    delta_in = (x - baseline)

    # DeepLIFT = (Δout / Δin) * Δin ≈ grad * Δin
    deeplift_attr = grads * delta_in
    return deeplift_attr.detach()


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


def ROAD(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    attr_func: Callable,
    # Argumentos da métrica (ROAD/AOPC):
    n_steps_metric: int = 100,  # Novo nome para não colidir com n_steps do IG
    percent_to_remove: float = 0.25,
    mask_value: float = 0.0,
    # Capturar argumentos obsoletos ou específicos do XAI:
    **kwargs: any
) -> float:
    """
    IMPLEMENTAÇÃO ROAD/AOPC (FIDELIDADE).
    Mede a queda na confiança do modelo quando as features mais relevantes são removidas 
    progressivamente. Um valor ALTO indica ALTA FIDELIDADE (a confiança cai muito).
    """
    if x.size(0) == 0:
        return 0.0

    # 1. Filtrar argumentos para a função de atribuição (attr_func)
    # Remove todos os argumentos que pertencem à métrica (ROAD) e não à atribuição (deeplift, etc.)
    attr_kwargs = {k: v for k, v in kwargs.items() if k not in ['n_samples', 'noise_std']}
    
    # 2. Preparar dados
    batch_size = x.size(0)
    
    # Calcular a atribuição (passando APENAS argumentos do XAI)
    attribution = attr_func(model, x, target, **attr_kwargs).squeeze(1).abs().flatten(1)
    
    # Ordenar os índices de atribuição do mais relevante para o menos relevante
    sorted_indices = torch.argsort(attribution, dim=1, descending=True)

    # Confiança original na classe alvo
    with torch.no_grad():
        output_orig = model(x).softmax(1)
        pred_orig = output_orig[torch.arange(batch_size), target]
    
    confidences = []
    x_perturbed = x.clone().to(x.device)
    
    # 3. Iteração e Perturbação (AOPC logic)
    
    total_pixels = attribution.size(1)
    max_pixels_to_remove = int(total_pixels * percent_to_remove)
    pixels_per_step = max_pixels_to_remove // n_steps_metric

    if pixels_per_step == 0:
        pixels_per_step = 1
        n_steps_metric = max_pixels_to_remove
    
    
    for i in range(1, n_steps_metric + 1):
        num_remove = i * pixels_per_step
        
        # Perturbar o input
        for b in range(batch_size):
            mask_indices = sorted_indices[b, :num_remove]
            x_perturbed.flatten(1)[b, mask_indices] = mask_value
        
        # 4. Calcular a nova confiança
        with torch.no_grad():
            output_pert = model(x_perturbed).softmax(1)
            pred_pert = output_pert[torch.arange(batch_size), target]
        
        # Queda de confiança (Fidelidade: Confiança Original - Confiança Perturbada)
        drop = (pred_orig - pred_pert).mean().item()
        confidences.append(drop)
        
        # Resetar o input
        x_perturbed = x.clone().to(x.device)
        
    # 5. O resultado é a média da queda de confiança (AOPC)
    return np.mean(confidences)


# === AVALIAÇÃO FINAL =======================================================

def evaluate_deeplift():
    print("\nIniciando avaliação com DeepLIFT...")
    model, x_batch, y_batch, device = load_model_and_data()

    # Cálculo inicial da atribuição
    attr = deeplift(model, x_batch, y_batch)

    # 1. CÁLCULO DAS MÉTRICAS
    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, deeplift, 
                     n_steps_metric=50, 
                     percent_to_remove=0.25) 
    }

    df = pd.DataFrame([results], index=["DeepLIFT"])
    csv_path = os.path.join(OUTPUT_DIR, "DeepLIFT_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)

    # === VISUALIZAÇÃO (3 LINHAS x 4 COLUNAS) =================================
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    label_to_show = 0
    fig.suptitle(f"Análise Visual das Métricas: DeepLIFT (Label {label_to_show})", fontsize=22, fontweight='bold', y=0.98)

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
    attr_noisy = deeplift(model, x_noisy, y_sample, baseline=None)
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
    plot_path = os.path.join(OUTPUT_DIR, f"DeepLIFT_Avaliacao_Label{label_to_show}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nVisualização salva em: {plot_path}")
    plt.close()
    
    return df

if __name__ == "__main__":
    evaluate_deeplift()