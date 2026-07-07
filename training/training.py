import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys

def loss_fn(): 
    loss = torch.nn.SmoothL1Loss()
    return loss 

def train(ldr,model,optimizer,device): 
    l1_loss_t = 0.0
    mae_loss_t = 0.0 
    model.train()
    n_batches = len(ldr)
    
    for batch in ldr: 
        maps = batch.groundtruth
        batch = batch.to(device)
        maps = maps.to(device)
        
        optimizer.zero_grad()
        
        pred_maps = model(batch)
        loss = loss_fn()(pred_maps,maps)
        loss.backward()
        optimizer.step()
        
        l1_loss_t += loss.item() 
        
        ### HERE SOMETHING WILL GO WRONG! WHAT DO WE WANT TO SUBTRACT FROM MAPS ? 
        mae_loss = torch.mean(abs(maps - pred_maps)).item()
        mae_loss_t += mae_loss
            
    return l1_loss_t / n_batches,  mae_loss_t / n_batches

def eval(ldr, model, device): 
    loss_t = 0.0 
    
    model.eval()
    
    for batch in ldr:
        maps = batch.groundtruth
        batch = batch.to(device)
        maps = maps.to(device)
        pred_maps = model(batch)
        
        mae_loss = torch.mean(abs(maps - pred_maps)).item()
        loss_t += mae_loss
        
    return loss_t /len(ldr)
         

def fit(ldr_train, ldr_test, model, epochs, device, exp_manager): 
    
    train_losses_l1 = []
    train_losses_mae = []
    val_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.to(device)

    for e in range(epochs): 
        print(f"Epoch {e}")
        
        train_loss_l1, train_loss_mae = train(ldr=ldr_train,
                           model=model,
                           optimizer=optimizer,
                           device=device)
        
        train_losses_l1.append(train_loss_l1)
        train_losses_mae.append(train_loss_mae)
        
        val_loss_mae = eval(ldr=ldr_test,
                            model=model,
                            device=device)
        
        val_losses.append(val_loss_mae)
        
        if exp_manager.step(val_loss_mae, model, e, optimizer,
                     train_loss_l1=train_loss_l1,
                     train_loss_mae=train_loss_mae): 
            print(f"[ExpManager] Early stopping at epoch {e} "
              f"(best {exp_manager.best_val_loss:.6f} @ {exp_manager.best_epoch})")   
            break     
        
        ### MISSING: SAVING BEST VERSION OF MODEL + Checkpoints 
        
    return model, train_losses_l1, train_losses_mae, val_losses
        
        
        
        
        
        
        
    
    
    