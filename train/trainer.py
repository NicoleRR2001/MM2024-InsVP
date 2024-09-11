# Desc 


import torch
import torchvision
from models import get_model
import time
from train.utils import device, get_dataset, cosine_lr
from train.scheduler import WarmupCosineSchedule as cosine_lr_2
import torch.optim as optim
import torch.nn as nn
import logging
import os
from torch.amp import autocast, GradScaler

from models.utils import Dataset_N_classes
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy




def train(args):
    """
    The training process
    """
    if args.loader == 'DAM-VP':
        from data_utils import loader as data_loader
    elif args.loader == 'E2VPT':
        from src.data import loader as data_loader
    logger_path = os.path.join(args.output_path, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log')  
    logging.basicConfig(filename=logger_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    model = get_model(args)
    model.mode = 'train'
    
    scaler = GradScaler()
    
    if args.mixup == 'mixup':
        criterion = SoftTargetCrossEntropy()
    else:   
        criterion = nn.CrossEntropyLoss()
           
    
    if args.optimizer == 'SGD':
        optimizer = model.get_optimizer()
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.learnable_parameters(), lr=args.lr, 
                            weight_decay=args.weight_decay)

    try:
        train_loader = data_loader.construct_train_loader(args, args.dataset)
    except:
        train_loader = data_loader.construct_train_loader(args)
    
    if args.scheduler == 'cosine-2':
        scheduler = cosine_lr_2(optimizer,
            len(train_loader)*args.warmup_epochs, 
            len(train_loader)*args.n_epochs
        )
    elif args.scheduler == 'cosine':
            scheduler = cosine_lr(optimizer, args.lr, 
                    len(train_loader)*args.n_epochs//5, 
                    len(train_loader)*args.n_epochs
                )

    if args.mixup == 'mixup':
        mixup_fn = Mixup(
        mixup_alpha=0.8,  
        cutmix_alpha=args.cutmix_alpha,  
        cutmix_minmax=None,  
        prob=0.5, 
        switch_prob=0.5,  
        mode='batch',  
        label_smoothing=0.1,  
        num_classes=Dataset_N_classes[args.dataset] 
        )

    accs = []
    best_acc = 0.0
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        model.train()
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            if args.scheduler == 'cosine':
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
            try:    
                images, labels = data['image'], data['label']
            except:
                images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            if args.mixup == 'mixup':
                if images.size(0) % 2 == 1:
                    images = images[:-1]
                    labels = labels[:-1]
                images, labels = mixup_fn(images, labels)
                
            with autocast(device_type='cuda'):
                outputs = model(images)        

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            try:
                correct += (predicted == labels).sum().item()
            except:
                ...
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
            logging.info(f"Epoch {epoch+1}, Iteration {i}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        print(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        logging.critical(f"Epoch {epoch+1} Finished, Loss: {running_loss / (i+1):.4f}, Accuracy: {correct / total:.4f}")
        
        acc = eval(model, args, epoch=epoch)
        accs.append(acc)
        
        if args.scheduler == 'cosine-2':
            scheduler.step()
          
        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(args.output_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            
        print(f"Best accuracy: {best_acc:.4f}")
        logging.critical(f"Best accuracy: {best_acc:.4f}")
        accs_tmp = [round(acc, 4) for acc in accs]
        print(f"All accuracy: {accs_tmp}")
        logging.critical(f"All accuracy: {accs_tmp}")  
        best_acc_tmp = round(best_acc, 4)
        with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
            f.write(f"Best accuracy:\n {best_acc_tmp}\n")
            f.write(f"All accuracy:\n {accs_tmp}\n")
            
    best_acc = round(best_acc, 4)
    accs = [round(acc, 4) for acc in accs]
        
    print("Finished Training")
    logging.critical("Finished Training")
    print(f"Best accuracy: {best_acc}")
    logging.critical(f"Best accuracy: {best_acc}")
    print(f"All accuracy: {accs}")
    logging.critical(f"All accuracy: {accs}")
    with open(os.path.join(args.output_path, 'accuracy.txt'), 'w') as f:
        f.write(f"Best accuracy:\n {best_acc}\n")
        f.write(f"All accuracy:\n {accs}\n")


    
def eval(model, args, epoch=0):
    """
    The evaluation process
    """
    if args.loader == 'DAM-VP':
        from data_utils import loader as data_loader
    elif args.loader == 'E2VPT':
        from src.data import loader as data_loader
    model.mode = 'test'
    model.eval()
    try:
        test_loader = data_loader.construct_test_loader(args, args.dataset)
    except:
        test_loader = data_loader.construct_test_loader(args)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            try:
                images, labels = data['image'], data['label']
            except:
                images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on the {total} test images:\n {correct / total:.4f}")
    logging.critical(f"Accuracy of the network on the {total} test images:\n {correct / total:.4f}")
    model.mode = 'train'
    return correct / total
            
    
    
