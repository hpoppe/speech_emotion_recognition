import os
import time
import h5py
import random 
import math
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch import optim 
import torch.onnx 
from torch.utils import data
from torch.utils.data import Dataset
import torchvision
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
import datetime
import librosa as lr
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold 
from vit_pytorch import ViT 
from torch.optim import SGD, lr_scheduler
from lion_pytorch import Lion 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import zscore 

from Architecture import cnnmodel1


def smooth_labels(y, factor):
    '''function by https://github.com/Jiaxin-Ye/TIM-Net_SER/blob/main/Code/Model.py'''
    assert 0 <= factor <= 1
    y *= 1 - factor
    y += factor / y.size(1)
    return y


class Data_Augmentation:
    def __init__(self, std=0.15):
        self.std = std

    def white_noise(self, spec_tensor):
        noise_tensor = self.std * torch.randn_like(spec_tensor)
        aug_spec_tensor = spec_tensor + noise_tensor
        return aug_spec_tensor
    
    
class DataLoader():
    def __init__(self, dataset_path, transform=None):
        self.dataset=h5py.File(dataset_path, "r", swmr=True)
        self.samples=list(self.dataset)
        self.dataset_len=len(self.samples)
        self.transform=transform
    
    def __getitem__(self, idx):
        i=self.samples[idx]
        self.name=self.samples[idx]
        self.data=self.dataset[i][:]
        if self.transform:
            self.data=self.transform(self.data)

        self.label=list(self.dataset[i].attrs.items())[0][1]
        return self.data, self.label, self.name
    
    def __len__(self):
        return self.dataset_len


#  Padding ESPnet https://github.com/espnet/espnet/tree/master
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):    
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)


def padding(batch):
    data_seq=[]
    label_seq=[]
    name_seq=[]
    mask_set_seq=[]
    for data, label, name in batch:
        data = np.array(data)
        data_seq.append(torch.tensor(data))
        mask_set_seq.append(torch.from_numpy(data).size(0))
        # print(data.shape)
        label_seq.append(torch.tensor([label]))
        name_seq.append(name)
    src_mask=make_non_pad_mask(mask_set_seq).unsqueeze(1)
    # print(f"src_mask: {src_mask.shape}")
    data_seq=pad_sequence(data_seq, batch_first=True, padding_value=0.0)
    label_seq=pad_sequence(label_seq, batch_first=True)
    return data_seq, label_seq, name_seq, src_mask


class Trainer():
    def __init__(self):
        self.best_valid_accuracy=None,
        self.best_model_state_dict=None,
        self.lda=None,
        self.model=None,
        self.confusion_matrix=None,
        self.eval_accuracy=None,
        self.save_model_path_pt=None,
        self.save_model_path_onnx=None,
        self.training_dataset=None,
        self.testing_dataset=None,
        self.train_subset=None,
        self.test_subset=None,
        self.valid_subset=None,
        self.optimizer=None,
        self.device=None,
        self.writer=None,
        self.criterion=None,
        self.scheduler=None,
        self.step_size=None,
        self.activateTensorboard=True


    def pitch_shift(self, audio, sr, n_steps):
        y_shifted = lr.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
        y_shifted /= np.amax(abs(y_shifted))
        return y_shifted


    def train(self, n_epochs, train_accu=[], eval_accu=[], sr=16000, num_classes=7, patience=60):
        train_accuracy=train_accu
        eval_accuracy=eval_accu
        idx=0
        lr_arr=[]
        counter = 0
        self.best_valid_accuracy=-np.inf
        for epoch in range(n_epochs):
            loss_train=0
            correct_train=0
            total_train=0
            accuracy_train=0
            t1 = time.time()
            self.model.train()
            pred_list = []
            target_list = []
            for i, (data_train, label_train, name, src_mask) in enumerate(self.training_dataset):         
                label_train = label_train.data.squeeze(1)
                data_=[data_train]
                for data_train in data_:
                    self.optimizer.zero_grad()
                    outputs_train=self.model(data_train, src_mask)
                label_train_one = F.one_hot(label_train, num_classes=num_classes).float()
                # label smoothing
                label_train_one=smooth_labels(label_train_one, 0.1)
                loss = self.criterion(outputs_train.float(), label_train_one.float()) 
                loss.backward()
                self.optimizer.step()

                loss_train += float(loss.item())
                _, prediction_train = torch.max(outputs_train.data, dim=1)
                total_train+=label_train.size(0)
                correct_train+=(prediction_train==label_train).sum().item()
                accuracy_train=100*correct_train/total_train
                accuracy_train=round(accuracy_train, 4)
                
                pred_list.extend(prediction_train.cpu().numpy())
                target_list.extend(label_train.cpu().numpy())

            if self.activateTensorboard:
                self.writer.add_scalar("Train Accuracy", accuracy_train, epoch*len(self.training_dataset)+i)
                self.writer.add_scalar("Train Loss", loss.item(), epoch*len(self.training_dataset)+i)
            idx+=1

            precision = precision_score(target_list, pred_list, average='weighted', zero_division=0)
            recall = recall_score(target_list, pred_list, average='weighted', zero_division=0)
            f1 = f1_score(target_list, pred_list, average='weighted', zero_division=0)            
            self.scheduler.step()
            lr_arr.append(self.optimizer.param_groups[0]["lr"])
            print(f'current lr: {self.optimizer.param_groups[0]["lr"]}')
            t2 = time.time()
            epoch_time=t2-t1
            m1, s1 = divmod(epoch_time, 60)
            print(f"Epoch [{epoch+1}/{n_epochs}], Time: {int(m1)}:{s1:.2f} minutes.")
            print(f"Train Accuracy = {accuracy_train}")
            print(f"Train Precision = {precision*100:.2f}")
            print(f"Train Recall = {recall*100:.2f}")
            print(f"Train F1-Score = {f1*100:.2f}")
            accuracy_valid, precision_eval, recall_eval, f1_eval, best_model = self.eval(epoch, self.validation_dataset)
        
            if accuracy_valid > self.best_valid_accuracy:
                counter=0
                self.best_model_state_dict = self.model.state_dict()
                self.best_valid_accuracy = accuracy_valid
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    print('Early stopping')
                    self.model.load_state_dict(torch.load('best_model.pt')) 
                    break
            precision_train=np.round(precision*100, decimals=4)
            recall_train=np.round(recall*100, decimals=4)
            f1_train=np.round(f1*100, decimals=4)
            train_accuracy.append([accuracy_train, precision_train, recall_train, f1_train])
            eval_accuracy.append([accuracy_valid, precision_eval, recall_eval, f1_eval])
            
        return train_accuracy, eval_accuracy, self.best_model_state_dict, lr_arr


    def eval(self, epoch, dataset_eval):
        correct_valid=0
        total_valid=0
        accuracy_valid=0
        self.model.eval()
        accuracy_eval_best=0
        best_model=0
        pred_list = []
        target_list = []
        with torch.no_grad():
            for i, (data_valid, label_valid, _, src_mask) in enumerate(self.validation_dataset):
                outputs_valid=self.model(data_valid, src_mask)
                outputs_valid = outputs_valid.to(torch.float32)
                label_valid=label_valid.squeeze(1)

                _, prediction_valid = torch.max(outputs_valid.data, 1)
                total_valid+=label_valid.size(0)
                correct_valid+=(prediction_valid==label_valid).sum().item()
                accuracy_valid=100*correct_valid/total_valid
                accuracy_valid=round(accuracy_valid, 4)

                pred_list.extend(prediction_valid.cpu().numpy())
                target_list.extend(label_valid.cpu().numpy())

                if self.activateTensorboard:
                    self.writer.add_scalar("Valid Accuracy", accuracy_valid)
            precision = precision_score(target_list, pred_list, average='weighted', zero_division=0)
            recall = recall_score(target_list, pred_list, average='weighted', zero_division=0)
            f1 = f1_score(target_list, pred_list, average='weighted', zero_division=0)
            print(f"Validation Accuracy = {accuracy_valid}")
            print(f"Validation Precision = {precision*100:.2f}")
            print(f"Validation Recall = {recall*100:.2f}")
            print(f"Validation F1-Score = {f1*100:.2f}")
            precision_eval=np.round(precision*100, decimals=4)
            recall_eval=np.round(recall*100, decimals=4)
            f1_eval=np.round(f1*100, decimals=4)
            return accuracy_valid, precision_eval, recall_eval, f1_eval, best_model


    def testing(self):
        correct_test=0
        total_test=0
        accuracy_test=0
        self.model.eval()
        all_labels = []
        all_predictions = []
        idx=0
        with torch.no_grad():
            for i, (data_test, label_test, _, src_mask) in enumerate(self.testing_dataset):
                self.model.load_state_dict(self.best_model_state_dict)
                outputs_test=self.model(data_test, src_mask)
                outputs_test = outputs_test.to(torch.float32)
                label_test = label_test.squeeze(1)

                _, prediction_test = torch.max(outputs_test.data, 1)
                all_labels.extend(label_test.tolist())
                all_predictions.extend(prediction_test.tolist())
                total_test+=label_test.size(0)
                correct_test+=(label_test==prediction_test).sum().item()
                accuracy_test=100*correct_test/total_test
                accuracy_test=round(accuracy_test, 4)
            cm = confusion_matrix(all_labels, all_predictions)
            label_names=["Wut", "Langeweile", "Ekel", "Angst", "Freude", "Trauer", "Neutral"]
            tick_marks = np.arange(len(label_names))
            font_settings = {
                'family': 'sans-serif',
                'size': 20,
                'weight': 'normal',
            }
            plt.figure(figsize=(20, 16))
            sns_heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', annot_kws={"size": 24}, cbar=False)
            plt.xlabel('Predicted', fontsize=24, fontweight='bold', labelpad=20)
            plt.ylabel('True', fontsize=24, fontweight='bold')
            plt.title('Berlin EMO-DB Confusion Matrix', fontsize=30, fontweight='bold', pad=20)
            plt.xticks(tick_marks + 0.5, label_names, ha='center', fontdict=font_settings)
            plt.yticks(tick_marks + 0.5, label_names, rotation=0, ha='right', fontdict=font_settings)
            cbar = plt.colorbar(sns_heatmap.collections[0], orientation='vertical')
            cbar.ax.tick_params(labelsize=20) 
            sns.despine(bottom=True, left=True)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            print(f"Test Accuracy: {accuracy_test}")
            print(f"Test Precision = {precision*100:.2f}")
            print(f"Test Recall = {recall*100:.2f}")
            print(f"Test F1-Score = {f1*100:.2f}")
            precision_test=np.round(precision*100, decimals=4)
            recall_test=np.round(recall*100, decimals=4)
            f1_test=np.round(f1*100, decimals=4)
            return accuracy_test, precision_test, recall_test, f1_test, cm

    

    def execute(self):
        # hyper parameters
        self.n_epochs=50
        self.batch_size=64
        self.learning_rate= 0.001
        self.train_size=0.9
        self.valid_size=0.1
        self.device = torch.device("cpu") # mps 
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        training_dataset_path = os.path.join(base_path, "Dataset/train_BerlinEMO_DB_mfcc_with_stats.h5")
        testing_dataset_path = os.path.join(base_path, "Dataset/test_BerlinEMO_DB_mfcc_with_stats.h5")
        valid_dataset_path = os.path.join(base_path, "Dataset/valid_BerlinEMO_DB_mfcc_with_stats.h5")

        self.training_dataset=DataLoader(training_dataset_path)
        self.testing_dataset=DataLoader(testing_dataset_path)
        self.validation_dataset=DataLoader(valid_dataset_path)

        print(f"Each feature vector is of shape: {self.training_dataset.__getitem__(0)[0].shape}")

        self.train_subset = self.training_dataset 
        self.valid_subset = self.validation_dataset 
        self.test_subset = self.testing_dataset

        self.training_dataset=data.DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)
        self.validation_dataset=data.DataLoader(self.valid_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)
        self.testing_dataset=data.DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)

        print(f"Train Size: {len(self.train_subset)}")
        print(f"Validation Size: {len(self.valid_subset)}")
        print(f"Test Size: {len(self.test_subset)}")

        features=40 
        classes=7

        self.model= cnnmodel1.cnn(features, classes)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=4e-5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.optimizer = optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=4e-5)
        # self.optimizer = Lion(self.model.parameters(), lr=self.learning_rate)
        
        num_training_steps = self.n_epochs * len(self.training_dataset)
        num_warmup_steps = 20
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )

        # self.scheduler=lr_scheduler.StepLR(self.optimizer, step_size=180, gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)

        time = datetime.datetime.now().strftime("%d-%m-%Y_%Hh:%M")
        txt_file = str(time)
        dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Training", time) + "/"
        self.writer = SummaryWriter(log_dir=dir_name)
        accuracy_train_total, accuracy_valid_total, best_model_state_dict, lr_arr = self.train(self.n_epochs)
        accuracy_test, precision_test, recall_test, f1_test, cm = self.testing()
        file_path=os.path.join(dir_name, txt_file)
        torch.save(self.model.state_dict(), os.path.join(dir_name, f"time_{time}.pt"))

        with open(f"{file_path}.txt", "w+") as file_spec:
            file_spec.write(f"Training Dataset: {os.path.basename(training_dataset_path)} \n\n")
            file_spec.write(f"Testing Dataset: {os.path.basename(testing_dataset_path)} \n\n")
            file_spec.write(f"Model: {repr(self.model)} \n\n")
            file_spec.write(f"Train Size: {self.train_size*len(self.training_dataset.dataset)} \n\n")
            file_spec.write(f"Valid Size: {self.valid_size*len(self.training_dataset.dataset)} \n\n")
            file_spec.write(f"Test Size: {len(self.testing_dataset.dataset)} \n\n")
            file_spec.write(f"Number of Epochs: {self.n_epochs} \n\n")
            file_spec.write(f"Learning Rate: {lr_arr} \n\n")
            file_spec.write(f"Accuracy, Precision, Recall, Test\n\n")
            file_spec.write(f"Train Accuracy: {accuracy_train_total} \n\n")
            file_spec.write(f"Valid Accuracy: {accuracy_valid_total} \n\n")
            file_spec.write(f"Test Accuracy: {[accuracy_test, precision_test, recall_test, f1_test]} \n\n")
            file_spec.close()


def main():
    trainer = Trainer()
    trainer.execute()

if __name__ == "__main__":
    main()