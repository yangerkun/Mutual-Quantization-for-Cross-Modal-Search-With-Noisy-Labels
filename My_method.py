from utils.tools import *
import itertools
from scipy.linalg import hadamard
from network import *
import pdb
import os
import torch
import torch.optim as optim
import time
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type = str, default = '1')
parser.add_argument('--hash_dim', type = int, default = 16)
parser.add_argument('--num_gradual', type = int, default = 20)
parser.add_argument('--noise_type', type = str, default = 'symmetric')
parser.add_argument('--noise_rate', type = float, default = 0.6)
parser.add_argument('--beta', type = float, default = 0.0)
parser.add_argument('--dataset', type = str, default = 'flickr')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

bit_len = args.hash_dim
noise_type = args.noise_type
noise_rate = args.noise_rate
dataset = args.dataset
beta = args.beta
num_gradual =  args.num_gradual

n_class = 0
tag_len = 0
if dataset == 'nuswide10':
    n_class = 10
    tag_len = 1000
elif dataset == 'flickr':
    n_class = 24
    tag_len = 1386
elif dataset == 'ms-coco':
    n_class = 80
    tag_len = 2000

torch.multiprocessing.set_sharing_strategy('file_system')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_config():
    config = {
        "gamma": 20.0,
        "alpha": 0.1,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "txt_optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": ResNet,
        "txt_net": TextNet,
        "dataset": dataset,
        "epoch": 100,
        "test_map": 2,
        "save_path": "save/DPSH",
        "device": torch.device("cuda:1"),
        "bit_len": bit_len,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "random_state": 1,
        "n_class": n_class,
        "lambda":0.0001,
        "topK": 5000,
        "tag_len":tag_len,
        "beta": beta
    }
    config = config_dataset(config)
    return config

class Robust_Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(Robust_Loss, self).__init__()
        self.is_single_label = config["dataset"] not in {"flickr", "nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to('cuda')
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to('cuda')
        self.criterion = torch.nn.BCELoss(reduction = 'none').to('cuda')

    def forward(self, u, v, y, ind, forget_rate, config):
        u = u.tanh()
        v = v.tanh()
        hash_center = self.label2center(y)
        center_loss1 = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        center_loss2 = self.criterion(0.5 * (v + 1), 0.5 * (hash_center + 1))
        Q_loss = (u.abs() - 1).pow(2) + (v.abs() - 1).pow(2)

        inner_product = torch.matmul(u, v.T)
        Softmax = torch.nn.Softmax(dim = 0)
        prob = Softmax(inner_product)
        p_prob = torch.diag(prob)
        MQ_loss = -1 * torch.log(p_prob)
        #loss =  center_loss + config["lambda"] * Q_loss + config["beta"] * kl
        loss =  center_loss1 + center_loss2 + config["lambda"] * Q_loss 
        loss = loss.mean(axis = 1)
        loss = loss + config["beta"] * MQ_loss
        cpu_loss = loss.data.cpu()
        ind_sorted = np.argsort(cpu_loss)
        #loss_sorted = loss[ind_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(cpu_loss))
        ind_update = ind_sorted[:num_remember]
        final_loss = torch.mean(loss[ind_update])
        
        return final_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets




class KLLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(KLLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to('cuda')
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to('cuda')

    def forward(self, u, v, ind, config):
        u = 0.5 * (u + 1)
        v = 0.5 * (v + 1)
        kl_1 = u*torch.log(torch.div(u,v))
        kl_2 = v*torch.log(torch.div(v,u))
        KL_loss = kl_1 + kl_2
        KL_loss = KL_loss.mean()
        return KL_loss



def train_val(config, bit):
    device = config["device"]
    train_loader,  test_loader, dataset_loader, num_train,  num_test, num_dataset = get_data(config)
    #pdb.set_trace()
    config["num_train"] = num_train
    net = config["net"](bit).to('cuda')
    txt_net = config["txt_net"](config['tag_len'], bit).to('cuda')
    parameters = itertools.chain(net.parameters(), txt_net.parameters())

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    txt_optimizer = config["txt_optimizer"]["type"](txt_net.parameters(), **(config["txt_optimizer"]["optim_params"]))

    criterion = Robust_Loss(config, bit)

    forget_rate = noise_rate
    rate_schedule = np.ones(config["epoch"]) * forget_rate
    rate_schedule[0:num_gradual] = np.linspace(0, forget_rate **1, num_gradual)
    #rate_schedule[0:40] = 0

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.eval()
        txt_net.eval()
        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            img_tst_binary, img_tst_tlabel, img_tst_label = compute_img_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            img_trn_binary, img_trn_tlabel, img_trn_label = compute_img_result(dataset_loader, net, device=device)
            txt_tst_binary, txt_tst_tlabel, txt_tst_label = compute_tag_result(test_loader, txt_net, device=device)
            txt_trn_binary, txt_trn_tlabel, txt_trn_label = compute_tag_result(dataset_loader, txt_net, device=device)
            img_tra_binary, img_tra_tlabel, img_tra_label = compute_img_result(train_loader, net, device=device)
            txt_tra_binary, txt_tra_tlabel, txt_tra_label = compute_tag_result(train_loader, txt_net, device=device)


            #dict_ = {'img_tst_binary': img_tst_binary, 'img_trn_binary': img_trn_binary, 'img_tra_binary': img_tra_binary,'img_tst_tlabel': img_tst_tlabel ,'img_trn_tlabel': img_trn_tlabel ,'img_tra_tlabel': img_tra_tlabel, 'img_tst_label': img_tst_label,'img_trn_label': img_trn_label,'img_tra_label': img_tra_label}
            #np.save('./npy_results/My_Robust_Flickr2_code_label_noiseType_{}_noiseRate_{}_beta_{}_epoch_{}.npy'.format(noise_type, noise_rate, beta, epoch),dict_)
            dict_ = {'img_tst_binary': img_tst_binary, 'img_trn_binary': img_trn_binary, 'img_tra_binary': img_tra_binary,'img_tst_tlabel': img_tst_tlabel ,'img_trn_tlabel': img_trn_tlabel ,'img_tra_tlabel': img_tra_tlabel, 'img_tst_label': img_tst_label,'img_trn_label': img_trn_label,'img_tra_label': img_tra_label,'txt_tst_binary': txt_tst_binary, 'txt_trn_binary': txt_trn_binary, 'txt_tra_binary': txt_tra_binary,'txt_tst_tlabel': txt_tst_tlabel ,'txt_trn_tlabel': txt_trn_tlabel ,'txt_tra_tlabel': txt_tra_tlabel, 'txt_tst_label': txt_tst_label,'txt_trn_label': txt_trn_label,'txt_tra_label': txt_tra_label}
            np.save('./npy_results/My_Robust_{}_code_{}_label_noiseType_{}_noiseRate_{}_epoch_{}.npy'.format(dataset, bit_len, noise_type, noise_rate, epoch),dict_)
            # print("calculating map.......")
            t2i_mAP = CalcTopMap(img_trn_binary.numpy(), txt_tst_binary.numpy(), img_trn_label.numpy(), txt_tst_label.numpy(),config["topK"])
            i2t_mAP = CalcTopMap(txt_trn_binary.numpy(), img_tst_binary.numpy(), txt_trn_label.numpy(), img_tst_label.numpy(),config["topK"])
            i2i_mAP = CalcTopMap(img_trn_binary.numpy(), img_tst_binary.numpy(), img_trn_label.numpy(), img_tst_label.numpy(),config["topK"])
            t2t_mAP = CalcTopMap(txt_trn_binary.numpy(), txt_tst_binary.numpy(), txt_trn_label.numpy(), txt_tst_label.numpy(),config["topK"])

            #cor_tr_t2i_mAP, oth_tr_t2i_mAP = TCalcTopMap(img_tra_binary.numpy(), txt_tra_binary.numpy(), img_tra_label.numpy(), txt_tra_label.numpy(),config["topK"], img_tra_tlabel.numpy(), txt_tra_tlabel.numpy())
            #cor_tr_i2t_mAP, oth_tr_i2t_mAP = TCalcTopMap(txt_tra_binary.numpy(), img_tra_binary.numpy(), txt_tra_label.numpy(), img_tra_label.numpy(),config["topK"], img_tra_tlabel.numpy(), img_tra_tlabel.numpy())
            #cor_tr_i2i_mAP, oth_tr_i2i_mAP = TCalcTopMap(img_tra_binary.numpy(), img_tra_binary.numpy(), img_tra_label.numpy(), img_tra_label.numpy(),config["topK"], img_tra_tlabel.numpy(), img_tra_tlabel.numpy())
            #cor_tr_t2t_mAP, oth_tr_t2t_mAP = TCalcTopMap(txt_tra_binary.numpy(), txt_tra_binary.numpy(), txt_tra_label.numpy(), txt_tra_label.numpy(),config["topK"], img_tra_tlabel.numpy(), txt_tra_tlabel.numpy())

            #print("%s epoch:%d, bit:%d, dataset:%s, t2i_mAP:%.3f, i2t_mAP:%.3f, i2i_mAP:%.3f, t2t_mAP:%.3f,cor_tr_t2i_mAP:%.3f, cor_tr_i2t_mAP:%.3f, cor_tr_i2i_mAP:%.3f, cor_tr_t2t_mAP:%.3f, oth_tr_t2i_mAP:%.3f, oth_tr_i2t_mAP:%.3f, oth_tr_i2i_mAP:%.3f, oth_tr_t2t_mAP:%.3f" % (
            #    config["info"], epoch + 1, bit, config["dataset"], t2i_mAP, i2t_mAP, i2i_mAP, t2t_mAP, cor_tr_t2i_mAP, cor_tr_i2t_mAP, cor_tr_i2i_mAP, cor_tr_t2t_mAP,oth_tr_t2i_mAP, oth_tr_i2t_mAP, oth_tr_i2i_mAP, oth_tr_t2t_mAP))
            print("%s epoch:%d, bit:%d, dataset:%s, t2i_mAP:%.3f, i2t_mAP:%.3f, i2i_mAP:%.3f, t2t_mAP:%.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], t2i_mAP, i2t_mAP, i2i_mAP, t2t_mAP))
            print(config)

        net.train()
        txt_net.train()

        train_loss = 0
        for image, tag, tlabel, label, ind in train_loader:
            image = image.to('cuda')
            tag = tag.to('cuda')
            tag = tag.float()
            label = label.to('cuda')

            optimizer.zero_grad()
            txt_optimizer.zero_grad()
            u = net(image)
            v = txt_net(tag)

           # loss1 = criterion1(u, label.float(), ind, config)
           # loss2 = criterion1(v, label.float(), ind, config)
            loss = criterion(u, v,label.float(), ind, rate_schedule[epoch], config)
            #loss = criterion(u, v,label.float(), ind, config)
           # loss = loss1 + loss2
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            txt_optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))



if __name__ == "__main__":
    config = get_config()
    print(config)
    #for bit in config["bit_list"]:
    train_val(config, config["bit_len"])
