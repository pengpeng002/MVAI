import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from validate import validate
from data import create_dataloader
from networks.vit_npr import create_model
from options.train_options import TrainOptions
from util import Logger
import torch.nn.functional as F
import random
import math
from torch.amp import autocast
# from time import time as tm
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class CLIPContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        logits = (image_features @ text_features.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, device=features.device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        similarity_matrix = similarity_matrix / self.temperature
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class VitTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = create_model(num_classes=1)
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.lr,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50
        )
        
        self.contrastive_criterion = ContrastiveLoss()
        self.CLIP_contrastive_criterion = CLIPContrastiveLoss()
        
        self.total_steps = 0
        self.lr = opt.lr
        self.scaler = torch.amp.GradScaler(device="cuda")
        
    def set_input(self, data):
        self.image = data[0].to(self.device)
        self.label = data[1].unsqueeze(1).float().to(self.device)     
        self.prompt_list = [
            "a photo of a real face" if l.item() < 0.5 else "a photo of a deepfake face"
            for l in self.label
        ]
        
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            output, features, text_features = self.model(self.image, self.prompt_list)
            output = torch.clamp(output, min=-20, max=20)
            cls_loss = self.criterion(output, self.label)
            contrastive_loss = self.contrastive_criterion(features)
            clip_contrastive_loss = self.CLIP_contrastive_criterion(features, text_features)
            self.loss = cls_loss + 0.01 * clip_contrastive_loss + 0.01 * contrastive_loss
            self.cls_loss = cls_loss
        print(f'losses: cls_loss={cls_loss}, clip_contrastive_loss={clip_contrastive_loss}, contrastive_loss={contrastive_loss}, total loss={self.loss}')
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def adjust_learning_rate(self):
        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        
    def save_networks(self, epoch):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.loss,
        }, save_path)
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt

class TopKManager:
    def __init__(self, k=5):
        self.k = k
        self.ranking = [] 
    
    def check_and_update(self, acc, new_file_path):
        if len(self.ranking) < self.k:
            self.ranking.append((acc, new_file_path))
            self.ranking.sort(key=lambda x: x[0], reverse=True) 
            return True
        min_acc, min_file_path = self.ranking[-1]
        if acc > min_acc:
            if os.path.exists(min_file_path):
                try:
                    os.remove(min_file_path)
                    print(f"[TopK] Removed low performance checkpoint: {os.path.basename(min_file_path)}")
                except OSError as e:
                    print(f"[TopK] Error deleting file: {e}")
            
            self.ranking.pop() 
            self.ranking.append((acc, new_file_path))
            self.ranking.sort(key=lambda x: x[0], reverse=True) 
            return True 
        return False

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best.pkl', trace_func=print, val_min=np.inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = val_min
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model_state_dict):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model_state_dict)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'[EarlyStopping] Patience: {self.counter} / {self.patience} (Best AUC: {self.best_score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True    
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model_state_dict)
            self.counter = 0

    def save_checkpoint(self, val_metric, model_state_dict):
        if self.verbose:
            self.trace_func(f'[EarlyStopping] Loss decreased ({self.val_min:.6f} --> {val_metric:.6f}). Saving model...')
        torch.save(model_state_dict, self.path)
        self.val_min = val_metric

if __name__ == '__main__':
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    opt = TrainOptions().parse()
    seed_torch(100)

    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))

    val_opt = get_val_opt()
    data_loader = create_dataloader(opt)
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    print(f"epoch number: {opt.niter}.")
    model = VitTrainer(opt)
    model.train()
    
    print("inited")
    load_epoch_num = 0
    start_epoch = load_epoch_num + 1
    val_loss=np.inf
    if load_epoch_num > 0:
        print('Loading checkpoint...')
        checkpoint = torch.load(f'./checkpoints/{opt.name}/model_epoch_{load_epoch_num}.pth', map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print('Loading from new checkpoint format...')
            state = checkpoint['model_state_dict']
            new_state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
            model.model.load_state_dict(new_state)
            print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
            if 'optimizer_state_dict' in checkpoint:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'Last loss: {checkpoint.get("loss", "unknown")}')
            val_loss = validate(model.model, val_opt)['val_loss']
            model.train()
            print(f"last val_loss = {val_loss}")
        else:
            print('Loading from old checkpoint format...')
            model.model.load_state_dict(checkpoint)

    print(f'cwd: {os.getcwd()}')
    orig_topk_manager = TopKManager(k=5)
    ckpt_dir = './checkpoints/save_for_es'
    save_path = os.path.join(ckpt_dir, "es_sace.pth")
    early_stopping = EarlyStopping(patience=6, verbose=True, path=save_path, val_min=val_loss)
    try:
        for epoch in range(start_epoch, opt.niter+1):
            epoch_start_time = time.time()
            epoch_iter = 0
            print(f'Starting epoch {epoch}/{opt.niter}...')
            for i, data in enumerate(data_loader):
                imgs_batch, labels_batch = data  
                model.set_input((imgs_batch, labels_batch))
                iter_data_time = time.time()
                model.total_steps += 1
                epoch_iter += opt.batch_size
                if i%400==0:
                    print(f'Processing batch {i+1}/{len(data_loader)}...')
                model.optimize_parameters()
                
                if model.total_steps % opt.loss_freq == 0:
                    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 
                        "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    train_writer.add_scalar('cls_loss', math.log10(model.cls_loss), model.total_steps)

            iter_end_time = time.time()
            print(f"Epoch {epoch} completed in {iter_end_time - epoch_start_time:.2f} seconds.")

            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 
                    'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()
            torch.cuda.empty_cache()

            model.eval()
            save_path = os.path.join(opt.checkpoints_dir, opt.name, f'model_epoch_orig_{epoch}.pth')
            res = validate(model.model, val_opt) #[:2]       
            acc, ap = res['acc'], res['ap']          
            val_loss = res['val_loss']
            early_stopping(val_loss, model.model.state_dict())
            val_writer.add_scalar('orig_accuracy', acc, epoch)
            val_writer.add_scalar('orig_ap', ap, epoch)
            print(f"orig: (Val Orig @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}")                
            if epoch >= 3 and orig_topk_manager.check_and_update(acc, save_path):
                model.save_networks(f'epoch_orig_{epoch}')
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}! No improvement for {early_stopping.patience} epochs.")
                break

            model.train()
            epoch_end_time = time.time()
            print(f"Epoch {epoch} total time: {epoch_end_time - epoch_start_time:.2f} seconds.")
    
        print("Training finished.")

    except KeyboardInterrupt:
        raise 
    except Exception as e:
        print("Training interrupted due to error!")
        raise e
