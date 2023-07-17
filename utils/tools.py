import numpy as np
import h5py
import pdb 
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
from noisyutils import *
import torchvision.datasets as dsets


def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class DataList(object):
    def __init__(self, dataset, data_type, transform, noise_type, noise_rate, random_state):
        self.data_type = data_type
        if dataset == 'nuswide10':
            data = h5py.File('/home/ekyang/Co-Quantization2/NUS-WIDE.h5', 'r')
        elif dataset == 'flickr':
            data = h5py.File('/home/ekyang/Co-Quantization2/MIRFlickr.h5', 'r')
        elif dataset == 'ms-coco':
            data = h5py.File('/home/ekyang/Co-Quantization2/MS-COCO.h5', 'r')
            #fi = h5py.File('/data/HashDatasets/NUS-WIDE/CVPR2019/Img-10.h5', 'r')
            #fl = h5py.File('/data/HashDatasets/NUS-WIDE/CVPR2019/Lab-10.h5', 'r')
            #ft = h5py.File('/data/HashDatasets/NUS-WIDE/CVPR2019/Tag-10.h5', 'r')
        if data_type == "train":
            fi = list(data['ImgTrain'])
            fl = list(data['LabTrain'])
            ffl = list(data['FLabTrain'])
            ft = list(data['TagTrain'])
            self.imgs = fi
            self.labs = fl
            self.flabs = ffl
            self.tags = ft
            lab = self.labs[1]
            lab = lab.astype(int)
            #pdb.set_trace()
            #lab = noisify(nb_classes=len(lab), train_labels=lab, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
        #    pdb.set_trace()
        #if data_type == "verify":
        #    fi = list(data['ImgVerify'])
        #    fl = list(data['LabVerify'])
        #    ft = list(data['TagVerify'])
        #    self.imgs = fi
        #    self.labs = fl
        #    self.tags = ft
        elif data_type == "test":
            fi = list(data['ImgQuery'])
            fl = list(data['LabQuery'])
            ft = list(data['TagQuery'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        elif data_type == "database":
            fi = list(data['ImgDataBase'])
            fl = list(data['LabDataBase'])
            ft = list(data['TagDataBase'])
            self.imgs = fi
            self.labs = fl
            self.tags = ft
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

    def __getitem__(self, index):
        img = self.imgs[index]
        #print(img.shape)
        img = Image.fromarray(img)
        #img = Image.open(path).convert('RGB')
        img = self.transform(img)
        lab = self.labs[index]
        lab = lab.astype(int)
        tlab = lab
        if self.data_type == "train":
            lab = self.flabs[index]
            lab = lab.astype(int)
        #pdb.set_trace()
            #lab, _ = noisify(nb_classes=len(lab), train_labels=lab, noise_type=self.noise_type, noise_rate=self.noise_rate, random_state=self.random_state)
        tag = self.tags[index]
        tag = tag.astype(float)
        #pdb.set_trace()
        return img, tag, tlab, lab, index

    def __len__(self):
        return len(self.imgs)

def SaveH5File(resize_size):
    #fi = h5py.File('/data/HashDatasets/NUS-WIDE/CVPR2019/Img-10.h5', 'r')
    fi = h5py.File('/data/HashDatasets/Flickr-25k/DCMH-Re/Img.h5', 'r')
    fl = h5py.File('/data/HashDatasets/Flickr-25k/DCMH-Re/Lab.h5', 'r')
    ft = h5py.File('/data/HashDatasets/Flickr-25k/DCMH-Re/Tag.h5', 'r')
    imgs = list(fi['ImgTrain'])
    labs = list(fl['LabTrain'])
    tags = list(ft['TagTrain'])
    n = len(imgs)
    Img = np.zeros([n,resize_size, resize_size, 3], dtype = np.uint8)
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in range(n):
        path = imgs[i]
        img_i = Image.open(path).convert('RGB')
        new_size = (resize_size, resize_size)
        img_i = img_i.resize(new_size)
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:,:,:] = img_i
        #pdb.set_trace()
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf = h5py.File('MIRFlickr.h5','w')
    hf.create_dataset('ImgTrain', data = Img)
    hf.create_dataset('TagTrain', data = Tag)
    hf.create_dataset('LabTrain', data = Lab)

    imgs = list(fi['ImgQuery'])
    labs = list(fl['LabQuery'])
    tags = list(ft['TagQuery'])
    n = len(imgs)
    Img = np.zeros([n,resize_size, resize_size, 3],  dtype = np.uint8)
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in range(n):
        path = imgs[i]
        img_i = Image.open(path).convert('RGB')
        new_size = (resize_size, resize_size)
        img_i = img_i.resize(new_size)
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:,:,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgQuery', data = Img)
    hf.create_dataset('TagQuery', data = Tag)
    hf.create_dataset('LabQuery', data = Lab)

    imgs = list(fi['ImgDataBase'])
    labs = list(fl['LabDataBase'])
    tags = list(ft['TagDataBase'])
    n = len(imgs)
    Img = np.zeros([n,resize_size, resize_size, 3],  dtype = np.uint8)
    Lab = np.zeros([n,24])
    Tag = np.zeros([n,1386])
    for i in range(n):
        path = imgs[i]
        img_i = Image.open(path).convert('RGB')
        new_size = (resize_size, resize_size)
        img_i = img_i.resize(new_size)
        img_i = np.asarray(img_i)
        lab_i = labs[i]
        lab_i = lab_i.astype(int)
        tag_i = tags[i]
        tag_i = tag_i.astype(float)
        Img[i,:,:,:] = img_i
        Tag[i,:] = tag_i
        Lab[i,:] = lab_i
    hf.create_dataset('ImgDataBase', data = Img)
    hf.create_dataset('TagDataBase', data = Tag)
    hf.create_dataset('LabDataBase', data = Lab)
    hf.close()

            
        


def image_transform(resize_size, crop_size, data_type):
    if data_type == "train":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])
#def pre_image_transform(resize_size, crop_size=224, data_type='train'):
#    step = [transforms.CenterCrop(crop_size)]
#    return transforms.Compose([transforms.Resize(resize_size)])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_type in ["train", "test", "database"]:
        dsets[data_type] = DataList(config["dataset"], data_type,
                                    image_transform(config["resize_size"], config["crop_size"], data_type), config["noise_type"], config["noise_rate"], config["random_state"])
        print(data_type, len(dsets[data_type]))
        dset_loaders[data_type] = util_data.DataLoader(dsets[data_type],
                                                      batch_size=config["batch_size"],
                                                      shuffle=True, num_workers=2)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"])


def compute_img_result(dataloader, net, device):
    bs, tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        bs.append((net(img.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(tclses), torch.cat(clses)

def compute_tag_result(dataloader, net, device):
    bs,tclses, clses = [], [], []
    net.eval()
    for img, tag, tcls, cls, _ in tqdm(dataloader):
        tclses.append(tcls)
        clses.append(cls)
        tag = tag.float()
        bs.append((net(tag.to('cuda'))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(tclses), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def TCalcTopMap(rB, qB, retrievalL, queryL, topk, tretrievalL, tqueryL):
    num_query = queryL.shape[0]
    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) > 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            #Cgnd = (np.dot(tqueryL[iter, :], tretrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]
            #cgnd = Cgnd[ind]

            tgnd = gnd[0:topk]
            #Ntgnd = Ngnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    cor_topkmap = topkmap / temp_ind

    topkmap = 0
    temp_ind = 0
    for iter in tqdm(range(num_query)):
        if np.dot(tqueryL[iter,:], queryL[iter,:].transpose()) == 0:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = CalcHammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            tgnd = gnd[0:topk]
            tsum = np.sum(tgnd).astype(int)
            if tsum == 0:
                continue
            count = np.linspace(1, tsum, tsum)

            tindex = np.asarray(np.where(tgnd == 1)) + 1.0
            topkmap_ = np.mean(count / (tindex))
            topkmap = topkmap + topkmap_
            temp_ind += 1
    oth_topkmap = topkmap / (temp_ind +0.0001)
    return cor_topkmap, oth_topkmap

if __name__ == "__main__":
    SaveH5File(256)
