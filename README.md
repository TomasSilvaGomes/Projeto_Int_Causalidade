# Projeto de Interpretabilidade e Causalidade

Este projeto explora métodos de interpretabilidade em redes neurais convolucionais (CNN) aplicadas ao dataset MNIST. O objetivo é comparar diferentes técnicas de XAI (Explainable AI) para entender como o modelo toma suas decisões.

## 📋 Descrição

O projeto implementa uma rede neural convolucional (CNN) em PyTorch para classificação de dígitos manuscritos do MNIST e aplica diversos métodos de interpretabilidade para visualizar as atribuições de importância dos pixels nas predições do modelo.

### Métodos de Interpretabilidade Implementados

- **Input X Gradient**: Multiplica os gradientes pelos valores de entrada
- **DeepLift**: Compara a ativação de cada neurônio com uma referência baseline
- **Integrated Gradients**: Calcula a integral dos gradientes ao longo do caminho da baseline até a entrada
- **Saliency**: Calcula o gradiente da saída em relação à entrada
- **Guided-GradCAM**: Combina Guided Backpropagation com Grad-CAM

## 🚀 Instalação

### Pré-requisitos

- Python 3.7 ou superior
- pip

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/TomasSilvaGomes/Projeto_Int_Causalidade.git
cd Projeto_Int_Causalidade
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
Projeto_Int_Causalidade/
├── Rede/
│   ├── rede_pytorch.py          # Implementação e treinamento da CNN
│   └── mnist_cnn_pytorch.pth    # Modelo treinado (gerado após o treino)
├── Metodos_interpretabilidade/
│   └── interpretability_methods.py  # Geração de visualizações XAI
├── Metricas/
│   ├── DeepLift.py              # Avaliação do método DeepLift
│   ├── GuidedGradCam.py         # Avaliação do método Guided-GradCAM
│   ├── InputXGradient.py        # Avaliação do método Input X Gradient
│   ├── Integrated Gradients.py  # Avaliação do método Integrated Gradients
│   └── Saliency.py              # Avaliação do método Saliency
├── Avaliacao_Metricas/          # Resultados das avaliações (CSV)
├── Imagens/                     # Visualizações geradas dos métodos XAI
├── sample_data/                 # Dataset MNIST (baixado automaticamente)
├── Tabela_Avaliacao.py          # Script para consolidar resultados
├── Tabela_Avaliacao.csv         # Tabela consolidada de avaliações
└── requirements.txt             # Dependências do projeto
```

## 🎯 Uso

### 1. Treinar o Modelo CNN

Para treinar o modelo (ou carregar um modelo já treinado):

```bash
python Rede/rede_pytorch.py
```

Este script:
- Baixa o dataset MNIST automaticamente (se necessário)
- Treina uma CNN por 10 épocas
- Salva o modelo em `Rede/mnist_cnn_pytorch.pth`
- Avalia a acurácia no conjunto de teste

### 2. Gerar Visualizações de Interpretabilidade

Para gerar as visualizações dos métodos XAI:

```bash
python Metodos_interpretabilidade/interpretability_methods.py
```

Este script:
- Carrega o modelo treinado
- Seleciona uma amostra de cada dígito (0-9)
- Aplica todos os métodos de interpretabilidade
- Salva as visualizações comparativas em `Imagens/`

### 3. Avaliar Métricas de Interpretabilidade

Para avaliar um método específico:

```bash
python Metricas/Saliency.py
python Metricas/DeepLift.py
# ... outros métodos
```

Cada script de métrica:
- Aplica o método de interpretabilidade
- Calcula métricas quantitativas usando o framework Quantus
- Salva os resultados em `Avaliacao_Metricas/`

### 4. Consolidar Resultados

Para criar uma tabela consolidada com todos os resultados:

```bash
python Tabela_Avaliacao.py
```

Gera o arquivo `Tabela_Avaliacao.csv` com todos os resultados de avaliação.

## 📊 Métricas de Avaliação

O projeto utiliza o framework [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) para avaliar quantitativamente os métodos de interpretabilidade. As métricas incluem:

- Fidelidade
- Robustez
- Complexidade
- Localização
- Entre outras

## 🛠️ Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning
- **Captum**: Biblioteca de interpretabilidade para PyTorch
- **Quantus**: Framework para avaliação de métodos XAI
- **NumPy & Pandas**: Manipulação de dados
- **Matplotlib**: Visualização
- **scikit-learn & scikit-image**: Processamento e análise

## 📝 Dependências Principais

```
torch==2.9.0+cu130
torchvision==0.24.0+cu130
captum==0.8.0
quantus==0.6.0
numpy==1.26.4
pandas==2.3.3
matplotlib==3.10.7
scikit-learn==1.7.2
```

Veja `requirements.txt` para a lista completa de dependências.

## 🖼️ Resultados

As visualizações geradas mostram como cada método de interpretabilidade destaca diferentes regiões da imagem como importantes para a predição do modelo. As imagens são salvas na pasta `Imagens/` com o formato:

- `Label0_Comparacao_XAI.png`
- `Label1_Comparacao_XAI.png`
- ... (para cada dígito de 0 a 9)

Cada imagem contém:
- A imagem original do dígito
- Visualizações de todos os métodos de interpretabilidade
- Colorbars indicando a intensidade da importância/contribuição

## 🔬 Arquitetura do Modelo

A CNN implementada possui a seguinte arquitetura:

```
Input: 1x28x28 (imagem MNIST)
Conv2D(1→10, kernel=5) + ReLU + MaxPool(2)
Conv2D(10→20, kernel=5) + ReLU + MaxPool(2)
Flatten
Linear(320→50) + ReLU
Linear(50→10)
Output: 10 classes (dígitos 0-9)
```

## 📄 Licença

Este projeto é de código aberto e está disponível para fins educacionais e de pesquisa.

## 👥 Autor

Tomás Silva Gomes

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
