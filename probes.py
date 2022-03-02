import torch
from torch import nn
from transformers import BertModel


class BaselineModel(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, num_class):
        super(BaselineModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.embedding.weight.requires_grad=False

        self.Linear = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        self.Linear.bias.data.zero_()
        
       
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output = self.Linear(embedded)
        
        return output

class TwoLayeredBaslineClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class,event_space=64,):
        super(TwoLayeredBaslineClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.embedding.weight.requires_grad=False

        self.Linear = nn.Linear(embed_dim, event_space)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(event_space, num_class)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        self.Linear.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Linear2.weight)
        self.Linear2.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        output = self.Linear(embedded)
        output = self.ReLU(output)
        output = self.Linear2(output)
        return output
        

class TwoLayeredBERTClassifier(nn.Module):

    def __init__(self, model_name, hidden_layer, num_class, event_space=64, embed_dim = 768):
        super(TwoLayeredBERTClassifier, self).__init__()
        self.Embeddings = BertModel.from_pretrained(model_name)
        self.Linear = nn.Linear(embed_dim, event_space)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(event_space, num_class)
        self.hidden_layer = hidden_layer
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        self.Linear.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Linear2.weight)
        self.Linear2.bias.data.zero_()


    def get_embeddings(self, text, offsets, cls=True):
        inputs = {'input_ids': text, 
                  'attention_mask': offsets}
        # output is hidden_state for each layer
        output = self.Embeddings(**inputs, output_hidden_states=True)
        #get hidden_state 
        hidden_states = output['hidden_states']
        if cls:
            hidden_layer = hidden_states[self.hidden_layer][:,:1,:]
        else:
            hidden_layer = hidden_states[self.hidden_layer][:,:,:]
        
        return torch.squeeze(hidden_layer) 

    def forward(self, text, offsets, cls=True):
        embedded = self.get_embeddings(text, offsets, cls)
        output = self.Linear(embedded)
        output = self.ReLU(output)
        output = self.Linear2(output)

        return output

class LinearBERTClassifier(nn.Module):

    def __init__(self, model_name, hidden_layer, num_class, embed_dim = 768):
        super(LinearBERTClassifier, self).__init__()
        self.Linear = nn.Linear(embed_dim, num_class)
        self.Embeddings = BertModel.from_pretrained(model_name)
        self.hidden_layer = hidden_layer
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        self.Linear.bias.data.zero_()

    def get_embeddings(self, text, offsets, cls=True):
        inputs = {'input_ids': text, 
                  'attention_mask': offsets}
        # output is hidden_state for each layer
        output = self.Embeddings(**inputs, output_hidden_states=True)
        #get hidden_state 
        hidden_states = output['hidden_states']

        if cls:
            hidden_layer = hidden_states[self.hidden_layer][:,:1,:]
        else:
            hidden_layer = hidden_states[self.hidden_layer][:,:,:]
        return torch.squeeze(hidden_layer) 


    def forward(self, text, offsets, cls=True):
        embedded = self.get_embeddings(text, offsets, cls=True)
        output = self.Linear(embedded)

        return output