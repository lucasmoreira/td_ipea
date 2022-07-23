# Text for Discussion Ipea (Institute for Applied Economic Research)

[TD 2612 - Línguas Naturais E Máquinas Artificiais: aplicação de técnicas de mineração de texto para a classificação de sentenças judiciais brasileiras](https://www.ipea.gov.br/portal/index.php?option=com_content&view=article&id=37082&Itemid=448)

This repository aims to use ML to classify sentence texts, using sample pre-sorted with regular expressions.

The files are sequenced as follows:

01_steam
Receives the sentence texts from the file trf2_td_1st.csv, with the case number, and returns the file data_stemized.csv with the data ready for analysis.

02_reverse_onehotencoding
Receives the texts cleaned in step 2 with the file: dados_stemizados.csv ,and the processes with sentence classification: tmp-sentencas-classificadas.csv. Then the data are merged into a single DF and the columns corresponding to the categorical classifications are converted to a single column. The final result is exported to the file: categorical.csv .

03_NB
the result of applying NB is run here, with confusion matrix.
