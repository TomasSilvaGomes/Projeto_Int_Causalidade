"""
interpretability_methods.py
Gera e visualiza atribuições XAI para a CNN em PyTorch, salvando o resultado.
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
import quantus
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from Rede.rede_pytorch import CNN, train_and_save_model
# Importações dos métodos Captum (necessário para o quantus.explain)
# O código anterior falhava sem estas importações explícitas
from captum.attr import InputXGradient, DeepLift, IntegratedGradients, Saliency, GuidedGradCam

# Constantes de Normalização do MNIST
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# =========================================================================
# 2. CORREÇÃO DE CONSTANTES: USAR CAMINHOS ABSOLUTOS BASEADOS NA RAIZ
# =========================================================================

# Configuração da Pasta
# OUTPUT_DIR será 'Projeto_Int_Causalidade/Imagens'
OUTPUT_DIR = os.path.join(project_root, "Imagens")
# MODEL_PATH será 'Projeto_Int_Causalidade/Rede/mnist_cnn_pytorch.pth'
MODEL_PATH = os.path.join(project_root, 'Rede', 'mnist_cnn_pytorch.pth')


def setup_data_and_model(model_path=MODEL_PATH):
    """Carrega o modelo PyTorch CNN e os dados de teste para explicação."""

    # 1. Assegurar que o modelo está treinado
    model = train_and_save_model(model_path=model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Modo de avaliação

    # 2. Preparar os dados de teste (com as transformações corretas)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./sample_data/MNIST', train=False, download=True, transform=transform)

    # 3. Selecionar uma imagem para cada label (0 a 9)
    x_list = []
    y_list = []

    # Encontra o primeiro exemplo para cada dígito de 0 a 9
    for label in range(10):
        # Encontra o índice da primeira ocorrência desta label
        idx = (test_dataset.targets == label).nonzero(as_tuple=True)[0][0].item()

        # Obtém a imagem (com a transformação aplicada)
        x_sample, y_sample = test_dataset[idx]

        # Salva o tensor e o rótulo
        x_list.append(x_sample)
        y_list.append(y_sample)

    # Converter para batch
    x_batch = torch.stack(x_list).to(device)
    y_batch = torch.tensor(y_list).to(device)

    # Retorna o modelo e o batch de 10 imagens (uma para cada label)
    return model, x_batch, y_batch, device


def generate_and_visualize_attributions(model, x_batch, y_batch, device):
    """Gera atribuições XAI e salva a visualização."""

    # Métodos XAI a serem usados (com sintaxe corrigida e melhorias)
    xai_methods = {
        "InputXGradient": {"method": "InputXGradient"},

        "DeepLift": {
            "method": "DeepLift",
            "n_samples": 100,
        },
        "Integrated Gradients": {
            "method": "IntegratedGradients",
            "baselines": torch.zeros_like(x_batch).to(device),
            "n_steps": 100,
        },
        "Saliency": {
            "method": "Saliency",
        },
        "Guided-GradCAM": {
            "method": "GuidedGradCam",
            "gc_layer": model.conv_layers[0],  # Altere esta linha
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nIniciando a geração e visualização para {len(x_batch)} amostras (uma para cada dígito).")

    # Itera sobre cada amostra (Label 0 a 9)
    for i in range(len(x_batch)):
        x_sample = x_batch[i].unsqueeze(0)  # Adiciona dimensão do batch
        x_sample.requires_grad_()
        y_label = y_batch[i].item()

        # Prepara a figura com 2 linhas: uma para imagens, outra para colorbars
        fig, axes = plt.subplots(
            2,
            len(xai_methods) + 1,
            figsize=(2.5 * (len(xai_methods) + 1), 3.5),
            gridspec_kw={'height_ratios': [1, 0.1]}  # Linha da colorbar mais fina
        )

        # 1. Imagem Original (Desnormalizada)
        # Inverte a normalização para visualização: Imagem = (Tensor * Std) + Mean
        mean = 0.1307
        std = 0.3081
        original_img = x_sample.detach().squeeze().numpy() * std + mean
        original_img = np.clip(original_img, 0, 1)  # Clipa valores

        ax_img = axes[0, 0]
        ax_img.imshow(original_img, cmap='gray')
        ax_img.set_title(f"Original (Label {y_label})", fontsize=10)
        ax_img.axis('off')
        axes[1, 0].axis('off')  # Desliga o eixo da colorbar para a imagem original

        # 2. Gera e plota as atribuições para todos os métodos
        for idx, (method_name, kwargs) in enumerate(xai_methods.items()):
            ax_img = axes[0, idx + 1]
            ax_cb = axes[1, idx + 1]

            try:
                # Usa quantus.explain para obter a atribuição
                attribution_batch = quantus.explain(
                    model=model,
                    inputs=x_sample,
                    targets=y_batch[i].unsqueeze(0),
                    device=device,
                    **kwargs
                )

                attribution_map = attribution_batch.squeeze()

                # Métodos que produzem apenas atribuições positivas ou cujo valor absoluto é mais informativo
                if method_name in ["InputXGradient", "DeepLiftShap", "Integrated Gradients", "Guided-GradCAM",
                                   "FusionGrad"]:
                    # Usar valor absoluto e um colormap sequencial para realçar a importância
                    abs_attr = np.abs(attribution_map)
                    im = ax_img.imshow(abs_attr, cmap="viridis")
                    fig.colorbar(im, cax=ax_cb, orientation='horizontal', label="Importância")

                # Métodos com atribuições positivas e negativas claras
                else:
                    # Manter o colormap divergente para ver contribuições positivas vs. negativas
                    max_val = np.abs(attribution_map).max()
                    im = ax_img.imshow(attribution_map, cmap="bwr", vmin=-max_val, vmax=max_val)
                    fig.colorbar(im, cax=ax_cb, orientation='horizontal', label="Contribuição")

                ax_img.set_title(method_name, fontsize=10)
                ax_img.axis('off')

            except Exception as e:
                ax_img.set_title(f"{method_name}\n(Erro)", fontsize=8)
                ax_img.axis('off')
                ax_cb.axis('off')  # Desliga também a colorbar em caso de erro
                print(f"Erro ao gerar {method_name} para Label {y_label}: {e}")

        # 3. Salvar a figura
        filename = os.path.join(OUTPUT_DIR, f"Label{y_label}_Comparacao_XAI.png")
        plt.tight_layout(h_pad=0.5)  # Ajusta o espaçamento vertical
        plt.savefig(filename)
        print(f"Salvo comparação XAI para Label {y_label} em: {filename}")
        plt.close(fig)


if __name__ == "__main__":
    try:
        # 1. Configurar
        model, x_batch, y_batch, device = setup_data_and_model()

        # 2. Gerar e Salvar Visualizações
        generate_and_visualize_attributions(model, x_batch, y_batch, device)

        print("\nProcesso de geração de imagens concluído com sucesso.")

    except Exception as e:
        print(f"\nOcorreu um erro fatal: {e}")