"""
guided_gradcam_evaluation.py
Avalia o método XAI Guided Grad-CAM na camada model.conv_layers[0]
com as métricas: Monotonicity, Sparseness e MaxSensitivity.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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


# === MÉTODO: GUIDED GRAD-CAM ==============================================

def guided_gradcam(model, x, target, layer):
    """
    Implementa o Guided Grad-CAM na camada especificada (por ex. model.conv_layers[0]).
    Combina Grad-CAM + Guided Backpropagation.
    """
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Registar hooks
    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_backward_hook(backward_hook)

    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    score = output[torch.arange(len(target)), target].sum()
    model.zero_grad()
    score.backward(retain_graph=False)

    # Obter gradientes e ativações
    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

    # Guided backpropagation
    guided_grad = torch.relu(x.grad.detach())

    # Combinar Guided BP e Grad-CAM
    guided_gradcam_map = guided_grad * cam
    handle_fwd.remove()
    handle_bwd.remove()

    return guided_gradcam_map.detach()


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


def ROAD(model, x, target, attr_func, n_samples=10, noise_std=0.5, layer=None):
    """
    Mede a robustez da explicação face a pequenas perturbações no input.
    Corresponde ao conceito de ROAD: Remove And Debias.
    """
    base_attr = attr_func(model, x, target, layer)
    max_diff = 0
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        noisy_attr = attr_func(model, x_noisy, target, layer)
        diff = torch.abs(base_attr - noisy_attr).mean().item()
        max_diff = max(max_diff, diff)
    return max_diff



# === AVALIAÇÃO FINAL =======================================================

def evaluate_guided_gradcam():
    print("\nIniciando avaliação com Guided Grad-CAM...")
    model, x_batch, y_batch, device = load_model_and_data()

    layer = model.conv_layers[0]
    attr = guided_gradcam(model, x_batch, y_batch, layer)

    results = {
        "Continuity": continuity(model, x_batch, y_batch, attr),
        "Selectivity": selectivity(model, x_batch, y_batch, attr),
        "ROAD": ROAD(model, x_batch, y_batch, guided_gradcam, n_samples=10, noise_std=0.5, layer= layer),
    }

    df = pd.DataFrame([results], index=["Guided Grad-CAM"])
    csv_path = os.path.join(OUTPUT_DIR, "GuidedGradCAM_Metricas.csv")
    df.to_csv(csv_path)
    print("\nResultados da Avaliação:")
    print(df)
    print(f"\nSalvo em: {csv_path}")
        # === VISUALIZAÇÃO DAS MÉTRICAS EM SUBPLOTS ================================
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))  # 3 métricas x 3 colunas (original, máscara, resultante)
    fig.suptitle("Avaliação Visual das Métricas para Guided Grad-CAM (Exemplo com Label 0)", fontsize=14)

    # Seleciona uma amostra (e.g., Label 0) para visualização
    sample_idx = 0
    x_sample = x_batch[sample_idx].unsqueeze(0)
    y_sample = y_batch[sample_idx].unsqueeze(0)
    attr_sample = attr[sample_idx].unsqueeze(0)

    # Função auxiliar para desnormalizar imagem
    def denormalize(img):
        return np.clip((img.cpu().numpy().squeeze() * MNIST_STD + MNIST_MEAN), 0, 1)

    # 1. Continuity
    ax = axes[0, 0]
    ax.imshow(denormalize(x_sample), cmap='gray')
    ax.set_title("Original (Continuity)")
    ax.axis('off')

    # Máscara cumulativa (exemplo: remove metade dos pixels mais importantes)
    flat_attr = attr_sample.flatten()
    sorted_indices = torch.argsort(flat_attr, descending=True)
    mask = torch.ones_like(flat_attr)
    mask[sorted_indices[:len(sorted_indices)//2]] = 0  # Remove top 50%
    mask = mask.view_as(x_sample)
    masked_x = x_sample * mask

    ax = axes[0, 1]
    ax.imshow(mask.squeeze().cpu().numpy(), cmap='hot', alpha=0.5)
    ax.set_title("Máscara Cumulativa")
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(denormalize(masked_x), cmap='gray')
    ax.set_title(f"Imagem Mascarada\n(Continuity: {results['Continuity']:.4f})")
    ax.axis('off')

    # 2. Selectivity
    ax = axes[1, 0]
    ax.imshow(denormalize(x_sample), cmap='gray')
    ax.set_title("Original (Selectivity)")
    ax.axis('off')

    # Máscara seletiva (remove top-10% das características)
    attr_flat = attr_sample.flatten(1)
    topk = int(0.1 * attr_flat.shape[1])  # Top 10%
    topk_indices = attr_flat.topk(topk, dim=1).indices
    mask_sel = torch.ones_like(attr_flat)
    mask_sel.scatter_(1, topk_indices, 0)
    masked_x_sel = (x_sample.flatten(1) * mask_sel).view_as(x_sample)

    ax = axes[1, 1]
    mask_vis = mask_sel.view_as(x_sample).squeeze().cpu().numpy()
    ax.imshow(mask_vis, cmap='hot', alpha=0.5)
    ax.set_title("Máscara Seletiva (Top 10%)")
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(denormalize(masked_x_sel), cmap='gray')
    ax.set_title(f"Imagem Perturbada\n(Selectivity: {results['Selectivity']:.4f})")
    ax.axis('off')

    # 3. ROAD
    ax = axes[2, 0]
    ax.imshow(denormalize(x_sample), cmap='gray')
    ax.set_title("Original (ROAD)")
    ax.axis('off')

    # Adiciona ruído e recalcula atribuição
    noise = torch.randn_like(x_sample) * 0.5
    x_noisy = x_sample + noise
    attr_noisy = guided_gradcam(model, x_noisy, y_sample, layer)

    ax = axes[2, 1]
    ax.imshow(denormalize(x_noisy), cmap='gray')
    ax.set_title("Imagem com Ruído")
    ax.axis('off')

    ax = axes[2, 2]
    diff_attr = torch.abs(attr_sample - attr_noisy).squeeze().cpu().numpy()
    ax.imshow(diff_attr, cmap='viridis')
    ax.set_title(f"Diferença nas Atribuições\n(ROAD: {results['ROAD']:.4f})")
    ax.axis('off')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "GuidedGradCAM_Visualizacao_Metricas.png")
    plt.savefig(plot_path)
    print(f"Visualização salva em: {plot_path}")
    plt.close()
    
    return df


if __name__ == "__main__":
    evaluate_guided_gradcam()

    
