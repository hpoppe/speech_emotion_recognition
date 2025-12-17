import os
import time
import h5py
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import torch 
from torch import nn
from torch import optim 
import torch.onnx
from torch.utils import data
from torch.optim import lr_scheduler
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import math
import random
import pyroomacoustics as pra
# from lion_pytorch import Lion 
from torchvision.transforms import RandomApply
import librosa as lr
from lion_pytorch import Lion
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from vit_pytorch import ViT
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import pearsonr

from Architecture import cnnmodel2

class WhiteNoiseAugmentation:
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, spec_tensor):
        noise_tensor = self.std * torch.randn_like(spec_tensor)
        aug_spec_tensor = spec_tensor + noise_tensor
        return aug_spec_tensor
    
# Padding by ESPnet https://github.com/espnet/espnet/tree/master
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


class DataLoader():
    def __init__(self, dataset_path, transform=None):
        self.dataset=h5py.File(dataset_path, "r", swmr=True)
        self.dataset_len=len(self.dataset)
        self.transform=transform
        self.samples=list(self.dataset)
    
    def __getitem__(self, idx):
        i=self.samples[idx]
        self.name=self.samples[idx]
        self.data=self.dataset[i][:]
        if self.transform:
            self.mfcc=self.transform(self.mfcc)
        self.arousal=list(self.dataset[i].attrs.items())[0][1]
        self.class_label=list(self.dataset[i].attrs.items())[1][1]
        self.dominance=list(self.dataset[i].attrs.items())[2][1]
        self.utterance=list(self.dataset[i].attrs.items())[3][1]
        self.valence=list(self.dataset[i].attrs.items())[4][1]
        return self.data, self.class_label, self.arousal, self.dominance, self.valence, self.utterance, self.name
    
    def __len__(self):
        return self.dataset_len






def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient. https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html"""
    # Raw data
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    # print(cor, sd_pred, sd_pred, var_pred, var_true, mean_pred, mean_true)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    if denominator == 0:
        raise ValueError("Denominator is zero.")
    return numerator / denominator


class Trainer():
    def __init__(self):
        self.model=None,
        self.confusion_matrix=None,
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
        self.criterion2=None,
        self.scheduler=None,
        self.step_size=None,
        self.activateTensorboard=True,
        self.min_length=None


    def pitch_shift(self, audio, sr, n_steps):
        y_shifted = lr.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
        y_shifted /= np.amax(abs(y_shifted))
        return y_shifted


    def train_valid_test_split(self):
        self.train_subset = self.training_dataset 
        self.valid_subset = self.validation_dataset 
        self.test_subset = self.testing_dataset


    def train(self, n_epochs, train_accu=[], eval_accu=[]):
        white_noise_transform = RandomApply([WhiteNoiseAugmentation()], p=0.5)
        train_accuracy=train_accu
        eval_accuracy=eval_accu
        m_length=self.min_length

        for epoch in range(n_epochs):
            loss_train=0
            correct_train=0
            total_train=0
            accuracy_train=0
            t1 = time.time()
            self.model.train()
            for i, (data_train, label_train, arousal_score, dominance_score, valence_score, utterance, name, src_mask) in enumerate(self.training_dataset):
                ground_truth=torch.from_numpy(np.concatenate((arousal_score, valence_score, dominance_score), axis=1))
                # print(data_train.shape)
                # label_train=torch.from_numpy(np.array(label_train)).long().to(self.device)
                data_train=torch.from_numpy(np.array(data_train)).double().to(self.device)
                self.optimizer.zero_grad()
                # print(data_train.shape, label_train.shape)
                label_train=label_train.squeeze(1)
                # print(data_train.shape)
                # label_train_one = F.one_hot(label_train, num_classes=9)
                # For transformerencoder/transformerencoderBERT, remove min_length parameter:
                # outputs_train=self.model(data_train.float(), src_mask, utterance)
                outputs_train=self.model(data_train.float(), src_mask, utterance, self.min_length)
                outputs_train=outputs_train.float()*6+1
                # print(outputs_train.shape)
                loss=self.criterion(outputs_train.float(),ground_truth.float())
                RMSE=np.sqrt(loss.detach().numpy())
                MAE=self.criterion2(outputs_train.float(),ground_truth.float())
                # print(ground_truth[:,1].float(), outputs_train[:,1].float())
                # print(ground_truth[:,1].detach().numpy(), outputs_train[:,1].detach().numpy())
                ccc_val = concordance_correlation_coefficient(ground_truth[:,1].detach().numpy(), outputs_train[:,1].detach().numpy())
                ccc_aro = concordance_correlation_coefficient(ground_truth[:,0].detach().numpy(), outputs_train[:,0].detach().numpy())
                ccc_dom = concordance_correlation_coefficient(ground_truth[:,2].detach().numpy(), outputs_train[:,2].detach().numpy())

                
                loss.backward()
                self.optimizer.step()
                # r2_train = r2_score(outputs_train.detach().numpy(), ground_truth.detach().numpy())            
            self.scheduler.step()
            t2 = time.time()
            epoch_time=t2-t1
            m1, s1 = divmod(epoch_time, 60)
            print(f"Epoch [{epoch+1}/{n_epochs}], Time: {int(m1)}:{s1:.2f} minutes.")
            print(f"Train CCC: {ccc_aro:.4f}, {ccc_val:.4f}, {ccc_dom:.4f}")
            ccc_value=[ccc_aro, ccc_val, ccc_dom]
            ccc_value_eval, MSE_eval, RMSE_eval, MAE_eval=self.eval(epoch, self.validation_dataset)
        return ccc_value, loss, RMSE, MAE, ccc_value_eval, MSE_eval, RMSE_eval, MAE_eval


    def eval(self, epoch, dataset_eval):
        correct_valid=0
        total_valid=0
        accuracy_valid=0
        self.model.eval()
        with torch.no_grad():
            for i, (data_valid, label_valid, arousal_score, dominance_score, valence_score, utterance, name, src_mask) in enumerate(self.validation_dataset):
                ground_truth=torch.from_numpy(np.concatenate((arousal_score, valence_score, dominance_score), axis=1))
                # label_valid=torch.from_numpy(np.array(label_valid)).long().to(self.device)
                # print(data_valid)
                data_valid=torch.from_numpy(np.array(data_valid)).double().to(self.device)
                # For transformerencoder/transformerencoderBERT, remove min_length parameter:
                # outputs_valid=self.model(data_valid.float(), src_mask, utterance)
                outputs_valid=self.model(data_valid.float(), src_mask, utterance, self.min_length)
                outputs_valid = outputs_valid.to(torch.float32)
                MSE=self.criterion2(outputs_valid.float(),ground_truth.float())
                RMSE=np.sqrt(MSE.detach().numpy())
                MAE=self.criterion2(outputs_valid.float(),ground_truth.float())
                # print(ground_truth[:,1].shape, outputs_valid[:,1])
                ccc_val = concordance_correlation_coefficient(ground_truth[:,1].detach().numpy(), outputs_valid[:,1].detach().numpy())
                ccc_aro = concordance_correlation_coefficient(ground_truth[:,0].detach().numpy(), outputs_valid[:,0].detach().numpy())
                ccc_dom = concordance_correlation_coefficient(ground_truth[:,2].detach().numpy(), outputs_valid[:,2].detach().numpy())
                # label_valid=label_valid.squeeze(1)
                # stats
                # r2_eval = r2_score(outputs_valid.detach().numpy(), ground_truth.detach().numpy())
                
                # print(r2_eval)
            print(f"Eval CCC: {ccc_aro:.4f}, {ccc_val:.4f}, {ccc_dom:.4f}")
            ccc_value=[ccc_aro, ccc_val, ccc_dom]
            return ccc_value, MSE, RMSE, MAE


    def testing(self):
        correct_test=0
        total_test=0
        accuracy_test=0
        self.model.eval()
        with torch.no_grad():
            for i, (data_test, label_test, arousal_score, dominance_score, valence_score, utterance, name, src_mask) in enumerate(self.testing_dataset):
                ground_truth=torch.from_numpy(np.concatenate((arousal_score, valence_score, dominance_score), axis=1))
                # label_test=torch.from_numpy(np.array(label_test)).long().to(self.device)
                data_test=torch.from_numpy(np.array(data_test)).double().to(self.device)
                # For transformerencoder/transformerencoderBERT, remove min_length parameter:
                # outputs_test=self.model(data_test.float(), src_mask, utterance)
                outputs_test=self.model(data_test.float(), src_mask, utterance, self.min_length)
                outputs_test = outputs_test.to(torch.float32)
                MSE=self.criterion2(outputs_test.float(),ground_truth.float())
                RMSE=np.sqrt(MSE.detach().numpy())
                MAE=self.criterion2(outputs_test.float(),ground_truth.float())
                # print(ground_truth.detach().numpy()[:,1], outputs_valid.detach().numpy()[:,1])
                ccc_val = concordance_correlation_coefficient(ground_truth[:,1].detach().numpy(), outputs_test[:,1].detach().numpy())
                ccc_aro = concordance_correlation_coefficient(ground_truth[:,0].detach().numpy(), outputs_test[:,0].detach().numpy())
                ccc_dom = concordance_correlation_coefficient(ground_truth[:,2].detach().numpy(), outputs_test[:,2].detach().numpy())

                # label_test = label_test.squeeze(1)
                # stats
                # r2_test = r2_score(outputs_test.detach().numpy(), ground_truth.detach().numpy())
        print(f"Test CCC: {ccc_aro:.4f}, {ccc_val:.4f}, {ccc_dom:.4f}")
        ccc_value=[ccc_aro, ccc_val, ccc_dom]
        return ccc_value, MSE, RMSE, MAE
    

    def execute(self):
        # hyper parameters
        self.n_epochs=10
        self.batch_size=64
        self.learning_rate=0.001
        self.train_size=0.9
        self.valid_size=0.1
        self.device = torch.device("cpu") # mps if you dont use weight normalization
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        training_dataset_path = os.path.join(base_path, "Dataset/train_MSP_dataset_mfcc_features_400_200.h5")
        testing_dataset_path = os.path.join(base_path, "Dataset/test_MSP_dataset_mfcc_features_400_200.h5")
        valid_dataset_path = os.path.join(base_path, "Dataset/valid_MSP_dataset_mfcc_features_400_200.h5")
        
        self.training_dataset=DataLoader(training_dataset_path)
        self.testing_dataset=DataLoader(testing_dataset_path)
        self.validation_dataset=DataLoader(valid_dataset_path)

        self.train_valid_test_split()
        
        # Calculate lengths for split
        total_len = len(self.training_dataset)
        train_len = int(total_len * self.train_size)
        valid_len = int(total_len * self.valid_size)
        test_len = total_len - train_len - valid_len
        
        # Generate random indices for disjoint split
        indices = list(range(total_len))
        random.shuffle(indices)
        
        train_indices = indices[:train_len]
        valid_indices = indices[train_len:train_len+valid_len]
        test_indices = indices[train_len+valid_len:]
        
        # Create disjoint subsets
        self.train_subset = torch.utils.data.Subset(self.training_dataset, train_indices)
        self.valid_subset = torch.utils.data.Subset(self.validation_dataset, valid_indices)
        self.test_subset = torch.utils.data.Subset(self.testing_dataset, test_indices)
    
        self.training_dataset=data.DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)
        self.validation_dataset=data.DataLoader(self.valid_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)
        self.testing_dataset=data.DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=True,drop_last=False, collate_fn=padding)

        print(f"Train Size: {len(self.train_subset)}")
        print(f"Validation Size: {len(self.valid_subset)}")
        print(f"Test Size: {len(self.test_subset)}")
        
        # Minimum length, gibt die kürzeste sequence length im Datensatz an
        self.min_length=200
        ''' to compute the min length of the sequence, computationally expensive.

        for i in range(0, self.train_subset.__len__()):
            if min_length>self.train_subset.__getitem__(i)[0][:].shape[0]:
                min_length=self.train_subset.__getitem__(i)[0][:].shape[0]
            # print(f"train: {min_length}")
        for i in range(0, self.valid_subset.__len__()):
            if min_length>self.valid_subset.__getitem__(i)[0][:].shape[0]:
                min_length=self.valid_subset.__getitem__(i)[0][:].shape[0]
            # print(f"vald:{min_length}")
        for i in range(0, self.test_subset.__len__()):
            if min_length>self.test_subset.__getitem__(i)[0][:].shape[0]:
                min_length=self.test_subset.__getitem__(i)[0][:].shape[0]
            # print(f"test:{min_length}")
        self.min_length=min_length
        print(self.min_length)'''
        
        number_features_MSP=39
        num_classes=3
        
        # ===== MODEL INITIALIZATION =====
        # To swap models, change the import at the top and the initialization below:
        # 
        # Option 1: cnnmodel2 (current)
        # Import: from Architecture import cnnmodel2
        # Init: self.model = cnnmodel2.cnn(self.min_length, number_features_MSP, num_classes)
        # Forward: self.model(data, src_mask, utterance, self.min_length)
        #
        # Option 2: transformerencoder
        # Import: from Architecture import transformerencoder
        # Init: self.model = transformerencoder.transformer_encoder(number_features_MSP, num_classes)
        # Forward: self.model(data, src_mask, utterance)  
        #
        # Option 3: transformerencoderBERT
        # Import: from Architecture import transformerencoderBERT
        # Init: self.model = transformerencoderBERT.transformer_encoder(number_features_MSP, num_classes)
        # Forward: self.model(data, src_mask, utterance) 
        # ================================
        
        self.model = cnnmodel2.cnn(self.min_length, number_features_MSP, num_classes)
        self.criterion = nn.MSELoss().to(self.device)
        self.criterion2 = nn.L1Loss().to(self.device)

        # if optimizer_adam:
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # self.optimizer = Lion(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)

        self.confusion_matrix=torch.zeros(3, 3)

        time = datetime.datetime.now().strftime("%d-%m-%Y_%Hh:%M")
        txt_file = str(time)
        dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", time) + "/"
        self.writer = SummaryWriter(log_dir=dir_name)
        
        CCC_train, MSE_train, RMSE_train, MAE_train, CCC_eval, MSE_eval, RMSE_eval, MAE_eval = self.train(self.n_epochs)

        CCC_test, MSE_test, RMSE_test, MAE_test = self.testing()

        file_path=os.path.join(os.getcwd(), (dir_name+txt_file))
        str_name=f'model_{txt_file}.pt'
        torch.save(self.model.state_dict(), str_name)
        with open(f"{file_path}.txt", "w+") as file_spec:
            file_spec.write(f"Training Dataset: {os.path.basename(training_dataset_path)} \n\n")
            file_spec.write(f"Testing Dataset: {os.path.basename(testing_dataset_path)} \n\n")
            file_spec.write(f"Model: {repr(self.model)} \n\n")
            file_spec.write(f"Train Size: {self.train_size*len(self.training_dataset.dataset)} \n\n")
            file_spec.write(f"Valid Size: {self.valid_size*len(self.training_dataset.dataset)} \n\n")
            file_spec.write(f"Test Size: {len(self.testing_dataset.dataset)} \n\n")
            file_spec.write(f"Number of Epochs: {self.n_epochs} \n\n")
            file_spec.write(f"Learning Rate: {self.learning_rate} \n\n")
            file_spec.write(f"Batch Size: {self.batch_size} \n\n")
            file_spec.write(f"CCC, MSE_train, RMSE_train, MAE_train: {CCC_train, MSE_train, RMSE_train, MAE_train} \n\n")
            file_spec.write(f"CCC, MSE_eval, RMSE_eval, MAE_eval: {CCC_eval, MSE_eval, RMSE_eval, MAE_eval} \n\n")
            file_spec.write(f"CCC, MSE_test, RMSE_test, MAE_test: {CCC_test, MSE_test, RMSE_test, MAE_test} \n\n")
            file_spec.close()


def main():
    trainer = Trainer()
    trainer.execute()

if __name__ == "__main__":
    main()