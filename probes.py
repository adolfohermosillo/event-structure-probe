import time
import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
from conllu import parse
from probes import BaselineModel,  TwoLayeredBaslineClassifier, TwoLayeredBERTClassifier, LinearBERTClassifier
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


argp = argparse.ArgumentParser()
argp.add_argument('model',
    help="Specify the model to use",
    choices=["baseline", "bert", "roberta"])

argp.add_argument('probe',
    help="Which probe to run ('Linear' or 'MLP')",
    choices=["Linear", "MLP"])

argp.add_argument('--enud_training_set_path',
    help="Path to the English Universal Dependencies training set", default=None)

argp.add_argument('--enud_dev_set_path',
    help="Path to the English Universal Dependencies development set", default=None)

argp.add_argument('--enud_test_set_path',
    help="Path to the English Universal Dependencies test set", default=None)

argp.add_argument('--event_structure_dataset',
    help="Path to the event structure dataset", default=None)

argp.add_argument('--model_name',
    help="Name of BERT or RoBERTa model", default=None)

argp.add_argument('--model_path',
    help="Path to save model", default=None)

argp.add_argument('--learning_rate',
    help="Set learning rate for optimizer", default=None)


argp.add_argument('--epochs',
    help="Set number of epochs", default=None)

argp.add_argument('--batch_size',
    help="Set batch size", default=None)

argp.add_argument('--layer',
    help="Set layer for BERT or RoBERTa", default=None)

args = parser.parse_args()



#compute F1
def evaluate(model, dataloader, criterion ):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc/total_count


def train_for_epoch(model, EPOCHS, train_dataloader, valid_dataloader, optimizer, criterion, scheduler, save_model):
    best_accuracy = 0
    total_accu = None
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer, criterion, scheduler)
        accu_val = evaluate(model, valid_dataloader, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
            if total_accu > best_accuracy:
                print('{} Accuracy --> {}'.format(best_accuracy, total_accu))
                best_accuracy = total_accu
                save_model_at = args.save_model + args.layer
                print('Saving model at ', save_model_at)
                torch.save(model.state_dict(),save_model_at)
                
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)
  
  
    
def train(model, dataloader, optimizer, criterion, scheduler):
    
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


            

# parsing the english universal dependencies currentyl in conll format
enud_train = open(args.enud_training_set_path,'r').read()
enud_dev = open(args.enud_dev_set_path,'r').read()
enud_test = open(args.enud_test_set_path,'r').read()

event_structure = pd.read_csv(args.event_structure_dataset, sep='\t')
event_structure_train = event_structure[event_structure.Split == 'train']
event_structure_dev = event_structure[event_structure.Split == 'dev']
event_structure_test = event_structure[event_structure.Split == 'test']

eng_ud_train = parse(enud_train)
eng_ud_dev = parse(enud_dev)
eng_ud_test = parse(enud_test)

train_set = get_data(distributive_train, eng_ud_train)
development_set = get_majority(get_data(distributive_dev, eng_ud_dev))
test_set = get_majority(get_data(distributive_test, eng_ud_test))



train_iter = iter(train_set)
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 768

if args.model == 'baseline':
    ###BUILDING BLOCKS BASELINE        
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) 
    COLLATE_FUNCTION = collate_batch
    
    if args.probe == 'Linear':
        model = BaselineModel(vocab_size, embed_dim, num_class)
        
    elif args.probe == 'MLP':
        model = TwoLayeredBaslineClassifier(vocab_size, embed_dim, num_class,event_space=64,)
        
    
elif args.model == 'bert':
    tokenizerBERT = BertTokenizer.from_pretrained(args.model_name)
    COLLATE_FUNCTION = collate_batchBERT
    
    if args.probe == 'Linear':
        model = LinearBERTClassifier(train_iter, num_class = num_class, event_space=64, embed_dim = 768) 
        
    elif args.probe == 'MLP':
        model = TwoLayeredBERTClassifier( train_iter , hidden_layer = args.layer, 
                                            num_class = num_class, event_space=64, embed_dim = 768) 
        
    
LR = args.learning_rate #learning rate
BATCH_SIZE = args.batch_size # batch size for training  
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)


train_dataset = to_map_style_dataset(iter(train_set))
dev_dataset = to_map_style_dataset(iter(development_set))
test_dataset = to_map_style_dataset(iter(test_iter))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=COLLATE_FUNCTION)
valid_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=COLLATE_FUNCTION)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=COLLATE_FUNCTION)

train_for_epoch(model = args.model_name, 
                EPOCHS = args.epochs,
                train_dataloader = train_dataloader,
                valid_dataloader = valid_dataloader, 
                optimizer = optimizer , 
                criterion = criterion, 
                scheduler = scheduler , 
                save_model = args.save_model)


