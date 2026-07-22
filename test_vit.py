import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.vit_npr import create_model
from options.test_options import TestOptions
import numpy as np
import random
import json
import pickle
from datetime import datetime
from thop import profile

def seed_torch(seed=1025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

DetectionTests = {
    'dataset1': { 
        'dataroot': './dataset/test',
        'no_resize': False,
        'no_crop': False,
    },
}

def save_results_comprehensive(results_dict, filename):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {filename}")

def save_results_binary(results_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"二进制结果已保存到: {filename}")

def test_model(model_path, batch_size=64, save_file=False, test_path=None):
    opt = TestOptions().parse(print_options=False)
    opt.model_path = model_path
    opt.batch_size = batch_size
    opt.save_file = save_file
    if test_path is not None:
        DetectionTests['dataset1']['dataroot'] = test_path
    print(f'Model_path: {opt.model_path}')
    print(f'Batch size: {opt.batch_size}')

    model = create_model(num_classes=1)

    print('Loading checkpoint...')
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print('Loading from new checkpoint format...')
        state = checkpoint['model_state_dict']
        new_state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
        model.load_state_dict(new_state)
        print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
        print(f'Last loss: {checkpoint.get("loss", "unknown")}')
    else:
        print('Loading from old checkpoint format...')
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"test_results_{timestamp}"
    
    for testSet in DetectionTests.keys():
        dataroot = DetectionTests[testSet]['dataroot']
        printSet(testSet)
        print('-' * 50)
        
        accs = []
        aps = []
        f1s = []
        rocs = []
        print(f'Testing started at: {time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}')
        test_results = {
            'test_set': testSet,
            'dataroot': dataroot,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'subsets': {}
        }

        for v_id, val in enumerate(sorted(os.listdir(dataroot))):
            opt.dataroot = os.path.join(dataroot, val)
            opt.classes = ''
            opt.no_resize = DetectionTests[testSet]['no_resize']
            opt.no_crop = DetectionTests[testSet]['no_crop']
            
            print(f'Testing on {val}...')
            res = validate(model, opt, False)
            test_results['subsets'][val] = res
            acc, ap, f1, roc = res['acc'], res['ap'], res['f1'], res['auc_roc']
            accs.append(acc)
            aps.append(ap)
            f1s.append(f1)
            rocs.append(roc)
            print(
                f"({v_id:2d} {val:12}) acc: {acc * 100:5.2f}; "
                f"ap: {ap * 100:5.2f}; f1: {f1 * 100:5.2f}; "
                f"auc_roc: {roc * 100:5.2f}"
            )
        
        mean_acc = np.array(accs).mean() * 100
        mean_ap = np.array(aps).mean() * 100
        mean_f1 = np.array(f1s).mean() * 100
        mean_roc = np.array(rocs).mean() * 100
        
        print(f"(Mean    ) mAcc: {mean_acc:5.2f}, mAP: {mean_ap:5.2f}, mF1: {mean_f1:5.2f}, mAuc_Roc: {mean_roc:5.2f}")
        print('*' * 50)

        if opt.save_file:
            os.makedirs(results_dir, exist_ok=True)
            json_filename = os.path.join(results_dir, f"results_{testSet}.json")
            # pickle_filename = os.path.join(results_dir, f"results_{testSet}.pkl")
            save_results_binary(test_results, pickle_filename)
            # save_results_comprehensive(test_results, json_filename)

        all_results[testSet] = {
            'accuracy': mean_acc,
            'ap': mean_ap,
            'f1': mean_f1,
            'auc_roc': mean_roc,
            'individual_accs': accs,
            'individual_aps': aps,
            'individual_f1s': f1s,
            'individual_aucs': rocs,
        }
        print(f'Testing ended at: {time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}')
    
    if len(DetectionTests.keys()) > 1:
        print("\nOverall Results Summary:")
        print('=' * 50)
        for testSet, results in all_results.items():
            print(f"{testSet:20} - Acc: {results['accuracy']:5.2f}, AP: {results['ap']:5.2f}")
        print('=' * 50)
    
    return all_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test ViT model on multiple datasets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test directory that contains dataset subsets')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--save_file', action='store_true', help='Save test results to file')

    args = parser.parse_args()
    
    test_model(args.model_path, args.batch_size, args.save_file, args.test_path) 