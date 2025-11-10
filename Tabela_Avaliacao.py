# juntar os csv todos da pasta Avaliacao_Metricas numa tabela so
import os
import pandas as pd

# Definir o caminho da pasta onde estão os arquivos CSV
pasta_avaliacao = 'Avaliacao_Metricas'
# Lista para armazenar os dataframes
dataframes = []
for arquivo in os.listdir(pasta_avaliacao):
    if arquivo.endswith('.csv'):
        caminho_arquivo = os.path.join(pasta_avaliacao, arquivo)
        df = pd.read_csv(caminho_arquivo)
        dataframes.append(df)
tabela_avaliacao = pd.concat(dataframes, ignore_index=True)
tabela_avaliacao.rename(columns={tabela_avaliacao.columns[0]: 'Modelo'}, inplace=True)

# Salvar o dataframe resultante em um novo arquivo CSV
tabela_avaliacao.to_csv('Tabela_Avaliacao.csv', index=False)
print("Tabela de avaliação criada com sucesso: Tabela_Avaliacao.csv")
