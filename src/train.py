import torch
import time
import sys
import os
import pickle
from src import dataset
from src.utils import save_model, load_model
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from src.logger import CSVLogger


           
def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    best_valid = float("inf")
    logger = CSVLogger(log_dir="results", run_name=hyp_params.name)


    def train_epoch(epoch):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for i_batch, (batch_X, batch_Y, _) in enumerate(train_loader):
            text, audio = batch_X
            labels = batch_Y.long() #here no need to flatten tensors as list of tensors available

            if hyp_params.use_cuda:
                text, audio, labels = text.cuda(), audio.cuda(), labels.cuda()

            optimizer.zero_grad()
            preds, _ = model(text, audio)
            loss = criterion(preds, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            epoch_loss += loss.item() * text.size(0)


        avg_loss = epoch_loss / hyp_params.n_train

        return avg_loss

    def evaluate(test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results, truths = [], []

        with torch.no_grad():
            for _, (batch_X, batch_Y, _) in enumerate(loader):
                text, audio = batch_X
                labels = batch_Y.view(-1).long()

                if hyp_params.use_cuda:
                    text, audio, labels = text.cuda(), audio.cuda(), labels.cuda() #force convert cpu tensors to cuda tensors 

                preds, _ = model(text, audio)
                total_loss += criterion(preds, labels).item() * text.size(0)
                results.append(preds)
                truths.append(labels)

        avg_loss = total_loss / len(loader.dataset)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train_epoch(epoch)
        val_loss, _, _ = evaluate(test=False)
        test_loss, test_preds, test_labels = evaluate(test=True)

        duration = time.time() - start
        scheduler.step(val_loss)
        
        acc = accuracy_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu())
        f1_w = f1_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu(), average='weighted')
        f1_m = f1_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu(), average='macro')

        print("-" * 50)
        print(f"Epoch {epoch:2d} | Time {duration:5.4f}s | "
              f"Train Loss {train_loss:5.4f} | Valid Loss {val_loss:5.4f} | Test Loss {test_loss:5.4f}")
        #print(f"Accuracy: {acc:.4f} | F1 (weighted): {f1_w:.4f} | F1 (macro): {f1_m:.4f}")
        print("-" * 50)
        
        logger.log(epoch, train_loss, val_loss, test_loss, acc, f1_w, f1_m)

        if val_loss < best_valid:
            print(f"Saved best model from epoch {epoch} at pretrained_models/{hyp_params.name}.pt!")
            save_model(model, name=hyp_params.name)
            best_valid = val_loss
        

    model = load_model(name=hyp_params.name,hyp_params=hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    _, test_preds, test_labels = evaluate(test=True)

    acc = accuracy_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu())
    f1_w = f1_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu(), average='weighted')
    f1_m = f1_score(test_labels.cpu(), test_preds.argmax(dim=1).cpu(), average='macro')

    print(f" Test Accuracy: {acc:.4f} | F1 (weighted): {f1_w:.4f} | F1 (macro): {f1_m:.4f}")
    
    logger.save_plot()

    return acc, f1_w, f1_m

