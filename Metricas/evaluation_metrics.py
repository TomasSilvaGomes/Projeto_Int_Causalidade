"""
evaluation_report.py
Avalia os métodos de interpretabilidade especificados pelo utilizador (InputXGradient, DeepLift, 
Integrated Gradients, FeatureAblation, Guided-GradCAM)
usando métricas de Fidelidade, Robustez e Sparsity no modelo PyTorch.

Gera uma tabela de resumo e um gráfico de radar.
"""
import sys
import os
# Adicionar a raiz do projeto ao sys.path para encontrar os outros módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import quantus
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from Rede.rede_pytorch import CNN, train_and_save_model
from collections import defaultdict

# --- Constantes ---
MODEL_PATH = 'Rede/mnist_cnn_pytorch.pth'
N_SAMPLES = 10

# --- Funções de Configuração ---

def setup_data_and_model(model_path=MODEL_PATH):
    """Carrega o modelo PyTorch CNN e os dados de teste para explicação."""
    model = train_and_save_model(model_path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./sample_data/MNIST', train=False, download=True, transform=transform)
    
    x_batch = torch.stack([test_dataset[i][0] for i in range(N_SAMPLES)]).to(device)
    y_batch = torch.tensor([test_dataset[i][1] for i in range(N_SAMPLES)]).to(device)
    
    return model, x_batch, y_batch, device

def define_evaluation_metrics():
    """Define as métricas de avaliação do Quantus: Selectivity, ROAD e Continuity."""
    return {
        "Selectivity": quantus.metrics.Selectivity(abs=False, normalise=False),
        "ROAD": quantus.metrics.ROAD(
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            abs=False,
            normalise=False
        ),
        "Continuity": quantus.metrics.Continuity(
            abs=False,
            normalise=False
        ),
    }

def define_xai_methods_for_evaluation(model, device):
    """Define os métodos XAI e suas configurações."""
    baseline_tensor = torch.zeros((1, 1, 28, 28)).to(device)
    target_layer = model.conv_layers[0]

    return {
        "InputXGradient": {"method": "InputXGradient"},
        "DeepLift": {"method": "DeepLift", "baselines": baseline_tensor},
        "IntegratedGradients": {"method": "IntegratedGradients", "baselines": baseline_tensor},
        "FeatureAblation": {"method": "FeatureAblation"},
        "GuidedGradCam": {"method": "GuidedGradCam", "gc_layer": target_layer},
    }

def predict_func_wrapper(model, device):
    """
    Cria um wrapper para a função de predição do modelo que garante a conversão
    correta entre NumPy arrays (usados por Quantus) e PyTorch Tensors (usados pelo modelo).
    """
    def predict(x_numpy):
        # Converte NumPy array para PyTorch Tensor
        x_tensor = torch.from_numpy(x_numpy).to(device)
        # Garante que o tensor tem o tipo de dados correto (float)
        if not x_tensor.is_floating_point():
            x_tensor = x_tensor.float()
        # Executa o modelo
        with torch.no_grad():
            outputs = model(x_tensor)
        # Converte a saída de volta para NumPy array
        return outputs.cpu().numpy()
    return predict


def generate_radar_chart(df_norm, methods):
    """Gera um gráfico de radar para a comparação visual."""
    categories = df_norm.index.tolist()
    N = len(categories)
    
    if N < 3:
        print("\nAVISO: Gráfico de radar não foi gerado (requer >= 3 métricas).")
        return

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) 
    
    plt.xticks(angles[:-1], categories, color='grey', size=12)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.5, 0.8, 1.0], ["0.2", "0.5", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    for method in methods:
        values = df_norm[method].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
        ax.fill(angles, values, alpha=0.1)

    plt.title("Comparação de Métodos XAI (Gráfico de Radar)", size=14, y=1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.95), fontsize=9)
    
    os.makedirs("Imagens", exist_ok=True)
    filename = "Imagens/Radar_Chart_XAI_Metrics.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"\nGráfico de radar salvo em: {filename}")

# --- Função Principal ---

def main():
    """
    Executa a avaliação completa de forma robusta, lidando com as
    inconsistências da API da versão do Quantus instalada.
    """
    print("--- Início da Avaliação Quantitativa (Modo Robusto Final) ---")
    
    # 1. Configurar ambiente
    model, x_batch, y_batch, device = setup_data_and_model()
    metrics_dict = define_evaluation_metrics()
    xai_methods_dict = define_xai_methods_for_evaluation(model, device)
    
    # 2. Loop de avaliação manual
    print(f"\nA avaliar {len(xai_methods_dict)} métodos em {N_SAMPLES} amostras...")
    results = defaultdict(dict)

    for method_name, method_params in xai_methods_dict.items():
        print(f"\n---\nAvaliar Método: {method_name}")
        
        # Gerar explicações em formato PyTorch Tensor
        a_batch = quantus.explain(model=model, inputs=x_batch, targets=y_batch, **method_params)

        for metric_name, metric_obj in metrics_dict.items():
            print(f"  - Calculando métrica: {metric_name}...")
            try:
                # Chamada simplificada. Passamos os dados como tensores,
                # que é o que a maioria das métricas espera.
                score = metric_obj(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=device
                )
                results[method_name][metric_name] = np.mean(score)
            except Exception as e:
                print(f"    ERRO ao calcular {metric_name} para {method_name}: {e}")
                results[method_name][metric_name] = np.nan

    # 3. Processar e exibir os resultados
    if not results:
        print("\nNenhum resultado foi calculado. A terminar.")
        return
        
    df_scores = pd.DataFrame(results).T  # Transpor para ter métodos nas linhas
    
    print("\n---------------------------------------------------------")
    print("--- Tabela de Resumo dos Scores (Brutos) ---")
    print("---------------------------------------------------------")
    print(df_scores.round(4))
    
    # 4. Normalizar os resultados
    df_norm = pd.DataFrame(index=df_scores.index)
    
    print("\n--- Tabela de Scores Normalizados (0=Pior, 1=Melhor) ---")
    print("---------------------------------------------------------")
    for metric_name, metric_func in metrics_dict.items():
        try:
            target_type = metric_func.target_type
        except AttributeError:
            target_type = getattr(metric_func, 'direction', 'max')

        scores = df_scores[metric_name].dropna()
        
        if scores.empty:
            df_norm[metric_name] = 0.5
            continue

        min_score, max_score = scores.min(), scores.max()
        
        if max_score == min_score:
            df_norm[metric_name] = 0.5 
        elif target_type == "max":
            df_norm[metric_name] = (scores - min_score) / (max_score - min_score)
        else: # target_type == "min"
            df_norm[metric_name] = (max_score - scores) / (max_score - min_score)

    print(df_norm.round(4))

    # 5. Gerar o gráfico de radar
    # CORREÇÃO: Adicionar .T para transpor o DataFrame
    generate_radar_chart(df_norm.T, list(xai_methods_dict.keys()))
    
    print("\n--- Fim da Avaliação ---")
    """
    Executa a avaliação completa de forma robusta, usando um predict_func_wrapper
    para garantir a compatibilidade de tipos e tratando cada métrica individualmente.
    """
    print("--- Início da Avaliação Quantitativa (Modo Robusto) ---")
    
    # 1. Configurar ambiente
    model, x_batch, y_batch, device = setup_data_and_model()
    metrics_dict = define_evaluation_metrics()
    xai_methods_dict = define_xai_methods_for_evaluation(model, device)
    
    # Criar o wrapper da função de predição
    predict_func = predict_func_wrapper(model, device)

    # 2. Loop de avaliação manual
    print(f"\nA avaliar {len(xai_methods_dict)} métodos em {N_SAMPLES} amostras...")
    results = defaultdict(dict)

    for method_name, method_params in xai_methods_dict.items():
        print(f"\n---\nAvaliar Método: {method_name}")
        
        # Criar a função de explicação para este método
        explain_func = lambda model, inputs, targets: quantus.explain(
            model=model, inputs=inputs, targets=targets, **method_params
        )

        # Gerar explicações uma vez, para as métricas que as usam diretamente
        a_batch = explain_func(model, x_batch, y_batch)

        for metric_name, metric_obj in metrics_dict.items():
            print(f"  - Calculando métrica: {metric_name}...")
            try:
                # Métricas como ROAD e Continuity precisam do explain_func.
                # Outras, como Selectivity, podem usar a_batch.
                # O try/except lida com as diferentes assinaturas.
                try:
                    # Tentativa 1: Para métricas que gerem as suas próprias explicações (ROAD, Continuity)
                    score = metric_obj(
                        model=model,
                        x_batch=x_batch.cpu().numpy(), # Passar como NumPy
                        y_batch=y_batch.cpu().numpy(),
                        explain_func=explain_func,
                        predict_func=predict_func,
                        device=device
                    )
                except TypeError:
                    # Tentativa 2: Para métricas que usam explicações pré-calculadas (Selectivity)
                    score = metric_obj(
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=a_batch,
                        predict_func=predict_func
                    )

                results[method_name][metric_name] = np.mean(score)
            except Exception as e:
                print(f"    ERRO ao calcular {metric_name} para {method_name}: {e}")
                results[method_name][metric_name] = np.nan

    # 3. Processar e exibir os resultados
    if not results:
        print("\nNenhum resultado foi calculado. A terminar.")
        return
        
    df_scores = pd.DataFrame(results).T  # Transpor para ter métodos nas linhas
    
    print("\n---------------------------------------------------------")
    print("--- Tabela de Resumo dos Scores (Brutos) ---")
    print("---------------------------------------------------------")
    print(df_scores.round(4))
    
    # 4. Normalizar os resultados
    df_norm = pd.DataFrame(index=df_scores.index)
    
    print("\n--- Tabela de Scores Normalizados (0=Pior, 1=Melhor) ---")
    print("---------------------------------------------------------")
    for metric_name, metric_func in metrics_dict.items():
        try:
            target_type = metric_func.target_type
        except AttributeError:
            target_type = getattr(metric_func, 'direction', 'max')

        # CORREÇÃO: Aceder à coluna corretamente
        scores = df_scores[metric_name].dropna()
        
        if scores.empty:
            df_norm[metric_name] = 0.5
            continue

        min_score, max_score = scores.min(), scores.max()
        
        if max_score == min_score:
            df_norm[metric_name] = 0.5 
        elif target_type == "max":
            df_norm[metric_name] = (scores - min_score) / (max_score - min_score)
        else: # target_type == "min"
            df_norm[metric_name] = (max_score - scores) / (max_score - min_score)

    print(df_norm.round(4))

    # 5. Gerar o gráfico de radar
    generate_radar_chart(df_norm, list(xai_methods_dict.keys()))
    
    print("\n--- Fim da Avaliação ---")

    """
    Executa a avaliação completa de forma manual, passando Tensores PyTorch
    diretamente para as métricas para evitar conflitos de tipo.
    """
    print("--- Início da Avaliação Quantitativa (Modo Manual Final) ---")
    
    # 1. Configurar ambiente
    model, x_batch, y_batch, device = setup_data_and_model()
    metrics_dict = define_evaluation_metrics()
    xai_methods_dict = define_xai_methods_for_evaluation(model, device)
    
    # 2. Loop de avaliação manual
    print(f"\nA avaliar {len(xai_methods_dict)} métodos em {N_SAMPLES} amostras...")
    results = defaultdict(dict)

    for method_name, method_params in xai_methods_dict.items():
        print(f"\n---\nAvaliar Método: {method_name}")
        
        a_batch = quantus.explain(model=model, inputs=x_batch, targets=y_batch, **method_params)
        
        for metric_name, metric_obj in metrics_dict.items():
            print(f"  - Calculando métrica: {metric_name}...")
            try:
                # Passar Tensores PyTorch diretamente para as métricas
                score = metric_obj(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch
                )
                results[metric_name][method_name] = np.mean(score)
            except Exception as e:
                print(f"    ERRO ao calcular {metric_name} para {method_name}: {e}")
                results[metric_name][method_name] = np.nan

    # 3. Processar e exibir os resultados
    df_scores = pd.DataFrame(results)
    
    print("\n---------------------------------------------------------")
    print("--- Tabela de Resumo dos Scores (Brutos) ---")
    print("---------------------------------------------------------")
    print(df_scores.round(4))
    
    # 4. Normalizar os resultados
    df_norm = pd.DataFrame(index=df_scores.columns)
    
    print("\n--- Tabela de Scores Normalizados (0=Pior, 1=Melhor) ---")
    print("---------------------------------------------------------")
    for metric_name, metric_func in metrics_dict.items():
        # CORREÇÃO: Obter o "alvo" da otimização da forma correta
        try:
            # Versão moderna
            target_type = metric_func.target_type
        except AttributeError:
            # Fallback para versões mais antigas
            target_type = getattr(metric_func, 'direction', 'max')

        scores = df_scores.loc[metric_name].dropna()
        
        if scores.empty:
            df_norm.loc[metric_name] = 0.5
            continue

        min_score, max_score = scores.min(), scores.max()
        
        if max_score == min_score:
            df_norm.loc[metric_name] = 0.5 
        elif target_type == "max":
            df_norm.loc[metric_name] = (scores - min_score) / (max_score - min_score)
        else: # target_type == "min"
            df_norm.loc[metric_name] = (max_score - scores) / (max_score - min_score)

    print(df_norm.round(4))

    # 5. Gerar o gráfico de radar
    generate_radar_chart(df_norm.T, list(xai_methods_dict.keys()))
    
    print("\n--- Fim da Avaliação ---")

if __name__ == "__main__":
    main()