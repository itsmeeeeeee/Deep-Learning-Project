import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from data_preprocessing import *


Xtrain = torch.tensor(X_train,dtype=torch.long)
Xtest = torch.tensor(X_test,dtype=torch.long)
ytrain = torch.tensor(y_train,dtype=torch.long)
ytest = torch.tensor(y_test, dtype=torch.long)
Xdev = torch.tensor(X_dev,dtype=torch.long)
ydev = torch.tensor(y_dev,dtype=torch.long)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

loader_train = data.DataLoader(data.TensorDataset(Xtrain, ytrain), shuffle=True, batch_size=4)
loader_dev = data.DataLoader(data.TensorDataset(Xdev, ydev), shuffle=True, batch_size=4)
loader_test = data.DataLoader(data.TensorDataset(Xtest, ytest), shuffle=True, batch_size=4)



class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.w1=nn.Linear(hidden_dim, hidden_dim)
        self.w2=nn.Linear(hidden_dim, hidden_dim)
        self.V=nn.Linear(hidden_dim,1)
    
    def forward (self, hidden, encoder_outputs):
        seq_len=encoder_outputs.shape[0]
        hidden=hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        
        scores=self.V(torch.tanh(self.w1(hidden)) + self.w2(encoder_outputs))
        attention=scores.squeeze(2)
        return F.softmax(attention,dim=-1)
        


class RNNBahdanauAttentionNER(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super(RNNBahdanauAttentionNER, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=2, dropout=0.5)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        outputs, hidden = self.rnn(embedded)
        seq_len, batch_size, _ = outputs.size()
        logits = []
        
        for i in range(seq_len):
            attention_weights = self.attention(hidden[1], outputs)
            context = torch.bmm(attention_weights.unsqueeze(1), outputs.permute(1, 0, 2)).squeeze(1)
            logit = self.fc(torch.cat((hidden[1], context), dim=1))
            logits.append(logit)
        logits = torch.stack(logits, dim=0)
        return logits

class EarlyStopping:
    def __init__(self, tolerance=3, min_delta= 0.1):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True




EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = TAG_COUNT




model_rnn=RNNBahdanauAttentionNER(len_uniq_words,EMBEDDING_DIM,HIDDEN_DIM,OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(model_rnn.parameters())


num_epochs=10
for epoch in range(num_epochs):
    model_rnn.train()
    train_loss=0.0
    #print(epoch)
    for x_batch, y_batch in loader_train:
        
        optimizer_rnn.zero_grad()
        tag_scores = model_rnn(x_batch)
        predictions=tag_scores.view(-1,tag_scores.shape[-1])
        
        tags=y_batch.view(-1)
        #print("true tag",tags)
        loss = criterion(predictions, tags)
        loss.backward()
        optimizer_rnn.step()
        train_loss += loss.item()
    train_loss/=len(loader_train)

        #Evaluation phase
    model_rnn.eval()
    dev_loss=0.0
    with torch.no_grad():
      for x_dev_batch,y_dev_batch in loader_dev:
        tag_scores = model_rnn(x_dev_batch)
        predictions=tag_scores.view(-1,tag_scores.shape[-1])
        
        tags=y_dev_batch.view(-1)
        loss = criterion(predictions, tags)

        dev_loss+=loss.item()
    dev_loss /= len(loader_dev)


    early_stopping = EarlyStopping(tolerance=3, min_delta= 0.1)


    early_stopping(train_loss, dev_loss)
    
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Dev Loss: {dev_loss}")
    if early_stopping.early_stop:
        print("We are at epoch:", epoch)
        break

def decode_tag(predictions,idtotag):
      decoded_tags=[idtotag[int(p)] for p in predictions]
      return decoded_tags


# Evaluate the model on test data
model_rnn.eval()
all_preds=[]
all_true_tags=[]

with torch.no_grad():
  for x_batch_test, y_batch_test in loader_test:
    tag_scores = model_rnn(x_batch_test)
    predictions=tag_scores.view(-1,tag_scores.shape[-1])
    max_pred=predictions.argmax(dim=-1)
    true_tags=y_batch_test.view(-1)
    decoded_pred=decode_tag(max_pred,id2tag)
    decoded_true_tag=decode_tag(true_tags,id2tag)
    all_preds.extend(decoded_pred)
    all_true_tags.extend(decoded_true_tag)
#print(all_preds)
#print(all_true_tags)
accuracy= accuracy_score(all_true_tags,all_preds)
precision = precision_score(all_true_tags, all_preds, average="weighted",zero_division=0)
#recall=recall_score(all_true_tags, all_preds)
#f1=f1_score(all_true_tags, all_preds)

print(f"Accuracy: {accuracy:.2f}")