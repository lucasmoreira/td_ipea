# TD Ipea



  Este repositório pretende usar ML para classificar textos de sentenças, utilizando amostra pré-classificada com expressões regulares.

  Os arquivos são sequenciados da seguinte forma:

## 01_steam

  Recebe os textos de sentença do arquivo trf2_td_1st.csv, com o número do processo, e retorna o arquivo dados_stemizados.csv com os dados prontos para análise.

## 02_reverse_onehotencoding

  Recebe os textos limpos no passo 2 com o arquivo: dados_stemizados.csv ,e os processos com sentença classificada: tmp-sentencas-classificadas.csv. Em seguida, são unidos os dados em um só DF e as colunas correspondentes às classificações categórigas são convertidas para uma só coluna.
  O resultado final é exportado para o arquivo: categorical.csv .
  
## 03_NB

  o resultado da aplicação do NB é executado aqui, com matriz de confusão.
