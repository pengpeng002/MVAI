import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from networks.vit_npr import create_model
from torch.amp import autocast
from data import create_dataloader
from copy import deepcopy
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix, 
                             roc_curve, precision_recall_curve)

def find_best_threshold(y_true, y_pred):
    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres

def metrics_calc(y_true, y_pred, overall=False):
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_pred)
    
    if overall==False:
        return {
            'acc': float(acc),
            'ap': float(ap),
            'f1': float(f1),
            'auc_roc': float(auc_roc)
        }
        
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    precision = precision_score(y_true, y_pred > 0.5, zero_division=0)
    recall = recall_score(y_true, y_pred > 0.5, zero_division=0)
    
    mu_real = y_pred[y_true == 0].mean() if (y_true == 0).any() else 0.0
    mu_fake = y_pred[y_true == 1].mean() if (y_true == 1).any() else 0.0
    std_real = float(y_pred[y_true == 0].std()) if (y_true == 0).any() else np.nan
    std_fake = float(y_pred[y_true == 1].std()) if (y_true == 1).any() else np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > 0.5).ravel()
    confusion_details = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    best_f1_idx = np.argmax(f1_scores)    
    if best_f1_idx < len(thresholds_pr):
        best_threshold = float(thresholds_pr[best_f1_idx])
    else:
        best_threshold = float(thresholds_pr[-1]) if len(thresholds_pr) > 0 else 0.5
    best_f1 = float(f1_scores[best_f1_idx])
    
    from sklearn.metrics import roc_curve
    fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, y_pred)
    
    detection_at_low_fpr = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr_curve - target_fpr))
        detection_at_low_fpr[f'tpr@fpr={target_fpr}'] = float(tpr_curve[idx])
        detection_at_low_fpr[f'threshold@fpr={target_fpr}'] = float(thresholds_roc[idx]) if idx < len(thresholds_roc) else np.nan

    high_conf_false_positives = int(((y_true == 0) & (y_pred > 0.9)).sum())
    high_conf_false_negatives = int(((y_true == 1) & (y_pred < 0.1)).sum())
    high_conf_errors = high_conf_false_positives + high_conf_false_negatives
    
    return {
        'acc': float(acc),
        'ap': float(ap),
        'r_acc': float(r_acc),
        'f_acc': float(f_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': confusion_details,
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'detection_at_low_fpr': detection_at_low_fpr,
        'high_conf_errors': {
            'false_positives': high_conf_false_positives,
            'false_negatives': high_conf_false_negatives,
            'total': high_conf_errors
        },
        'mu_real': float(mu_real),
        'std_real': float(std_real),
        'mu_fake': float(mu_fake),
        'std_fake': float(std_fake)

    }

def validate(model, opt, valLoss=True):
    data_loader = create_dataloader(opt)

    with autocast(device_type="cuda"):
        with torch.no_grad():
            y_true, y_pred = [], []
            if valLoss:
                total_cls_loss = 0.0
                criterion = torch.nn.BCEWithLogitsLoss()
            for i, data in enumerate(data_loader):
                imgs_batch, labels_batch = data 
                in_tens = imgs_batch.cuda()
                labels_tens = labels_batch.cuda().float()
                output = model(in_tens)
                pred = output.sigmoid().flatten().tolist()
                if valLoss:
                    loss = criterion(output.flatten(), labels_tens.flatten())
                    total_cls_loss += loss.item()
                y_pred.extend(pred)
                y_true.extend(labels_batch.flatten().tolist())
         
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ret = metrics_calc(y_true, y_pred)
    if valLoss:
        avg_val_loss = total_cls_loss / len(data_loader)   
        ret['val_loss']= avg_val_loss
    return ret
