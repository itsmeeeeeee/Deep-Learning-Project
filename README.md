# Deep-Learning-Project
# BiLSTM und RNN-Architektur mit dem Bahdanau Attention für NER-Aufgabe

Im Projekt werden zwei Deep-Learning-Modelle im PyTorch-Framework trainiert, basierend auf dem Kaggle NER-Dataset. Es handelt sich um ein BiLSTM-Modell und ein RNN-Modell mit Bahdanau Attention. 

# Usage
* BiLSTM-Modell trainieren
  ```
  python3 BiLSTM.py
  ```
* RNN-Modell mit Bahdanau Attention trainieren
  
  ```
  python3 RNNAttention.py
  ```
* Data Preprocessing 
  
  ```
  python3 data_preprocessing.py
  ```

# Requirements
* Python 3.11.3
* Pytorch 1.12.0
* Pandas 2.0.2
* Numpy 1.25.0
* Sklearn 1.2.2

# Training Results BiLSTM
![Alt text](/Bilstm_ergebnisse.png?raw=true "BiLSTM Training")

# Training Results RNNAttention Training
![Alt text](/Ergebnisse_RNNAttention.png?raw=true "RNNAttention Training")
