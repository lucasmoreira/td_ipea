#reverte os resultados de one_hot_encoding para uma coluna categoriga

import numpy as np
import pandas as pd

#importando classificacao dos processos
dataset = pd.read_csv('tmp-sentencas-classificadas.csv', delimiter = '\t')
#importando texto stemizado
dataset_td = pd.read_csv('dados_stemizados.csv', delimiter = ';')
#unindo resultado dos DF pela coluna id de processo
result = pd.concat([dataset, dataset_td["texto_movimento_novo"]], axis=1, sort= False, join="inner")
#fatiando apenas classificacao onehotencoding
encoded_data = result.loc[:,"acordo":"extincao"]
encoded_data = encoded_data.values

#funcao para retornar numero equivalente categorigo
def decode(datum):
    return np.argmax(datum)

#salva qual o numero corresponde a classificacao em lista
categorical = []
for i in range(encoded_data.shape[0]):
    decoded_datum = decode(encoded_data[i])
    categorical.append(decoded_datum)

result["categorical"] = categorical

result.to_csv("categorical.csv")
