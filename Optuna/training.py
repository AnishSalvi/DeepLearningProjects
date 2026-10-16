# install file

#imports
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #":16:8"
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.utils.data import default_collate
torchvision.disable_beta_transforms_warning()
import torch.nn.functional as F
import numpy as np
import random 
import transformers
import sklearn
import json
import datetime
from datetime import datetime
import pandas as pd
import multiprocessing
import subprocess
import numbers
import platform
import re
import warnings
import statistics
import math
import collections
import optuna
import timm
from functools import partial
import warnings
import sys
warnings.filterwarnings('ignore')
#set cpu count
cpu_count = multiprocessing.cpu_count() - 1
print('CPU Count', cpu_count)

#set torch
if torch.cuda.is_available():
    device = 'cuda'
    gpu = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    print('Device: ', gpu)
    print('Num GPU: ', gpu_count)
else:
    device = 'cpu'
    gpu = None
    gpu_count = 0
    print('Device', device)

#get micromamba version
def get_micromamba_version():
    try:
        #runs micromamba --version // decodes the string output
        result = subprocess.run(['micromamba', '--version'], capture_output = True, text = True, check = True)
        version = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        version = 'Micromamba is not installed or not in your system PATH.'
    return version

#python version
def get_python_version():
    """
    Returns the current Python interpreter's version.
    """
    return sys.version

#cudnn for reproducibility
def cudnn_(cudnn = False, deterministic = True, benchmark = False):
    #control whether cuDNN is enabled --> set to False to reproduce // should over-ride following two commands
    torch.backends.cudnn.enabled = cudnn
    #deterministically set algorithim, reduces performance if set to True --> set to True to reproduce
    torch.backends.cudnn.deterministic = deterministic
    #deterministic based on machine type, can boost performance if set to True  --> set to False to reproduce
    torch.backends.cudnn.benchmark = benchmark
    #specific algorithims
    torch.use_deterministic_algorithms(True)
    #uninit memory
    torch.utils.deterministic.fill_uninitialized_memory = True
    #debug mode
    torch.set_deterministic_debug_mode(2)

#seeding
def seeding(seed = 42):
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    np.random.seed(seed)
    random.seed(seed)

#processor name
def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

#get the model
def get_model(model_config):
    #pretrained path
    if model_config['pretrained_path']:
        #get masked autoencoder vit
        model = MaskedAutoencoderViT(pretrained_path = model_config['pretrained_path'], 
                                     #image_size = model_config['image_size'],
                                     #patch_size = model_config['patch_size'],
                                     mask_ratio = model_config['mask_ratio'],
                                     num_channels = model_config['in_channels'], 
                                     latent_dim = model_config['latent_dim'], 
                                     encoder_attention_dropout = model_config['encoder_attention_dropout'], 
                                     encoder_output_dropout = model_config['encoder_output_dropout'],
                                     decoder_attention_dropout = model_config['decoder_attention_dropout'],
                                     decoder_output_dropout = model_config['decoder_output_dropout'],
                                     interpolate_pos_encoding = model_config['interpolate_pos_encoding'],
                                     masked_loss = model_config['masked_loss'],
                                    )
    return model

#init_weights
def weights_init(net):
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            torch.nn.init.zeros_(module.bias.data)

#set dropout
def set_dropout(model, 
                encoder_attention_dropout = 0.1, 
                encoder_output_dropout = 0.2,
                decoder_attention_dropout = 0.3,
                decoder_output_dropout = 0.4
               ):
    #iterate
    for name, module in model.named_modules():
        #isinstance
        if isinstance(module, torch.nn.Dropout):
            #encoder attention dropout
            if ('encoder' in name) and ('attention' in name):
                module.p = encoder_attention_dropout
            #encoder output dropout
            if ('encoder' in name) and ('attention' not in name):
                module.p = encoder_output_dropout
            #decoder attention dropout
            if ('decoder' in name) and ('attention' in name):
                module.p = decoder_attention_dropout
            #decoder output dropout
            if ('decoder' in name) and ('attention' not in name):
                module.p = decoder_output_dropout

#set dropout
def set_encoder_dropout(encoder, 
                        encoder_embedding_dropout = 0.1,
                encoder_attention_dropout = 0.2, 
                encoder_output_dropout = 0.3,
               ):
    #iterate
    for name, module in encoder.named_modules():
        #isinstance
        if isinstance(module, torch.nn.Dropout):
            #embedding dropout
            if 'embeddings' in name:
                module.p = encoder_embedding_dropout
            #encoder attention dropout
            if ('encoder' in name) and ('attention' in name):
                module.p = encoder_attention_dropout
            #encoder output dropout
            if ('encoder' in name) and ('attention' not in name):
                module.p = encoder_output_dropout

#unfreeze positional embeddings
def unfreeze_pos(model):
    #iterate
    #for name, module in model.named_modules():
    for param in model.parameters():
        param.requires_grad = True
    #return model

#MaskedAutoEncoderViT
class MaskedAutoencoderViT(torch.nn.Module):
    #init
    def __init__(self, pretrained_path = 'facebook/vit-mae-huge', #image_size = 224, patch_size = 14, 
                 mask_ratio = 0.5, num_channels = 1, 
                 encoder_attention_dropout = 0.1, 
                 encoder_output_dropout = 0.1,
                 decoder_attention_dropout = 0.1,
                 decoder_output_dropout = 0.1,
                 interpolate_pos_encoding = False, 
                 latent_dim = 768, 
                 masked_loss = False
                ):
        #super
        super(MaskedAutoencoderViT, self).__init__()
        #config
        config = transformers.ViTMAEConfig.from_pretrained(pretrained_path)
        #set config
        #config.image_size = image_size
        #config.patch_size = patch_size
        #config.mask_ratio = mask_ratio
        #config.num_channels = num_channels
        #config.attention_probs_dropout_prob = attention_probs_dropout_prob
        #config.hidden_dropout_prob = hidden_dropout_prob
        #model
        self.model = transformers.ViTMAEForPreTraining.from_pretrained(pretrained_path, 
                                                                       force_download = True, 
                                                                       from_tf = False, 
                                                                       ignore_mismatched_sizes = True,
                                                                       attn_implementation = 'sdpa',
                                                                       config = config)
        #unfreeze 
        unfreeze_pos(self.model)
         #masked loss
        self.masked_loss = masked_loss
        #set dropout
        set_dropout(self.model, 
                    encoder_attention_dropout = encoder_attention_dropout, 
                    encoder_output_dropout = encoder_output_dropout,
                    decoder_attention_dropout = decoder_attention_dropout,
                    decoder_output_dropout = decoder_output_dropout)
        #change model input
        self.ch_model(in_channels = num_channels)
        #change model output
        self.lin_model(num_channels = num_channels)
        #mask_ratio
        self.model.config.mask_ratio = mask_ratio
        #interpolate for larger sizes
        self.interpolate_pos_encoding = interpolate_pos_encoding
        #classifier
        #self.classifier = torch.nn.Linear(self.model.config.hidden_size,
        #                                num_labels)
        #init weights
        #self.classifier.apply(weights_init)
        #norm layer
        self.layer_norm = torch.nn.LayerNorm(self.model.config.hidden_size)
        #linear layer
        self.lat_linear = torch.nn.Linear(self.model.config.hidden_size, latent_dim)
        #init weights
        self.lat_linear.apply(weights_init)
    
    #forward cls loss
    def forward_cls_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true)
    
    #all recontruction losses
    def mse_loss(self, pixel_values, pred, interpolate_pos_encoding):
        #patchify
        target = self.model.patchify(pixel_values, interpolate_pos_encoding)
        #pix_loss
        if self.model.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        #compute
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean() #take a mean over all patches
        return loss
    
    #compute hidden
    def compute_hidden(self, latent):
        #average pooling
        hidden = latent[:, 1:, :].mean(dim = 1)
        #norm
        hidden = self.layer_norm(hidden)
        #return
        return self.lat_linear(hidden)
    
    #modify model by channel
    def ch_model(self, in_channels = 1):
        #projection weight
        proj_weight = self.model.vit.embeddings.patch_embeddings.projection
        #new weight
        m = torch.nn.Conv2d(
              in_channels = in_channels,
              out_channels = proj_weight.out_channels,
              kernel_size = proj_weight.kernel_size,
              stride = proj_weight.stride
          )
        #copy
        m.weight = torch.nn.Parameter(proj_weight.weight[:, 0:in_channels, ])
        #copy
        m.bias = torch.nn.Parameter(proj_weight.bias)
        #weight
        self.model.vit.embeddings.patch_embeddings.projection = m
        #num_channels
        self.model.config.num_channels = in_channels
        #num_channels
        self.model.vit.embeddings.patch_embeddings.num_channels = in_channels
        
    #modify model by linear
    def lin_model(self, num_channels = 1):
        #set
        self.model.decoder.config.num_channels = num_channels
        #linear weight
        linear_weight = self.model.decoder.decoder_pred
        #new weight
        m = torch.nn.Linear(
            in_features = self.model.config.decoder_hidden_size,
            out_features = self.model.decoder.config.patch_size ** 2 * self.model.decoder.config.num_channels,
            bias = True
            )
        #copy weight
        m.weight = torch.nn.Parameter(linear_weight.weight[:m.out_features, :])
        #copy bias
        m.bias = torch.nn.Parameter(linear_weight.bias[:m.out_features])
        #insert
        self.model.decoder.decoder_pred = m
        
    #forward
    def forward(self, x_true):
        #pass through vit
        outputs = self.model.vit(x_true, interpolate_pos_encoding = self.interpolate_pos_encoding)
        #pass through decoder
        decoder_outputs = self.model.decoder(outputs.last_hidden_state, outputs.ids_restore, 
                                             interpolate_pos_encoding = self.interpolate_pos_encoding)
        #compute hidden
        hidden = self.compute_hidden(outputs.last_hidden_state[:, 1:, :])
        #pass through loss for reconstruction
        if self.masked_loss:
            rec_loss = self.model.forward_loss(x_true, decoder_outputs.logits, outputs.mask,
                                     interpolate_pos_encoding = self.interpolate_pos_encoding)
        else:
            #mean squared error loss                                   
            rec_loss = self.mse_loss(x_true, decoder_outputs.logits, 
                                     interpolate_pos_encoding = self.interpolate_pos_encoding)
        #get classification using the cls_token
        #y_pred = self.classifier(outputs.last_hidden_state[:, 0, :])
        #compute cls_loss
        #cls_loss = self.forward_cls_loss(y_pred, y_true)
        #return loss, logits, last hidden state with cls token removed,
        #return outputs.last_hidden_state[:, 1:, :], decoder_outputs.logits, rec_loss, y_pred, cls_loss
        return hidden, decoder_outputs.logits, rec_loss

#load json file
def load_params(fpath):
    with open(fpath) as json_file:
        data = json.load(json_file)
    return data

#save json file
def save_params(hyper_params, save_path):
    json_string = json.dumps(hyper_params)
    with open(save_path, 'w') as outfile:
        outfile.write(json_string)
        
#cutout
class Cutout():
    def __init__(self, scale = (0.05, 0.05), ratio = (0.1, 3)):
        self.cutout = torchvision.transforms.v2.RandomErasing(p = 1, scale = scale, ratio = ratio)
        #self.num_transforms = [
            #torchvision.transforms.v2.RandomHorizontalFlip(p = 1),
            #torchvision.transforms.v2.RandomVerticalFlip(p = 1),
            #torchvision.transforms.v2.RandomRotation(degrees = 180)
        #]
    def __call__(self, image):
        mask = torch.ones(image.shape)
        mask = self.cutout(mask)
        inv_mask = torch.abs(mask - 1)
        #apply
        mask = mask * image
        #apply
        inv_mask = inv_mask * image * torch.normal(image.mean(), image.std(), size = image.shape)
        #inv_mask = inv_mask * image * torchvision.transforms.v2.functional.gaussian_blur(image, kernel_size = (3, 3), sigma = 25)
        #inv_mask = inv_mask * image * torchvision.transforms.functional.gaussian_blur(image, kernel_size = (3, 3), sigma = 25)
        #torch.normal(image.mean(), image.std(), size= image.shape)
        #add
        return mask + inv_mask

#mirroir
class Mirroir():
    def __init__(self, img_h = 512, img_w = 512, shift_h = 0.2, shift_w = 0.2):
        self.img_h = img_h
        self.img_w = img_w
        self.shift_h = self.img_h * shift_h
        self.shift_w = self.img_w * shift_w
    def __call__(self, img):
        shift_h = int(self.shift_h * random.random())
        shift_w = int(self.shift_w * random.random())
        if random.random() > 0.5:
            out = torch.concatenate([img[:,:,shift_w:],
                               img[:,:,:shift_w]], axis = 2)
        else:
            out = torch.concatenate([img[:,:,int(self.img_w - shift_w):],
                               img[:,:,:int(self.img_w - shift_w)]], axis = 2)
        if random.random() > 0.5:
            out = torch.concatenate([out[:,shift_w:,:],
                                   out[:,:shift_w,:]], axis = 1)
        else:
            out = torch.concatenate([out[:,int(self.img_w - shift_w):,:],
                                   out[:,:int(self.img_w - shift_w),:]], axis = 1)
        return out

#patchify
class Patchify():
    def __init__(self, img_h = 299, img_w = 299, patch_h = 3, patch_w = 3):
        self.patch_h = img_h // patch_h
        self.patch_w = img_w // patch_w
    def __call__(self, img): #channelxheightxwidth
        ls_h = list(torch.split(img, self.patch_h, dim = 1))
        random.shuffle(ls_h)
        img = torch.concatenate(ls_h, dim = 1)
        ls_w = list(torch.split(img, self.patch_w, dim = 2))
        random.shuffle(ls_w)
        return torch.concatenate(ls_w, dim = 2)

#apply transform
def apply_transform(img, p, transform, transforms):
    #check if transform should be applied
    if p > random.random():
        if transform is None:
            aug = random.randint(1, len(transforms))
        else:
            aug = transform
        #collection of augmentation
        img_transforms = random.sample(transforms, aug)
        #apply the select transforms
        for img_transform in img_transforms:
            img = img_transform(img)
    #return
    return img

#custom augmentation
class Custom_Augment():
    #init
    def __init__(self, aug_p = 1.0, aug_transform = None, glb_p = 1.0, glb_transform = None, 
                 loc_p = 1.0, loc_transform = None, height = 299, width = 299, shift_h = 0.25, shift_w = 0.25, 
                 patch_h = 2, patch_w = 2):
        #aug_p
        self.aug_p = aug_p
        #aug transform
        self.aug_transform = aug_transform
        #glb_p
        self.glb_p = glb_p
        #glb transform
        self.glb_transform = glb_transform
        #loc_p
        self.loc_p = loc_p
        #loc transform
        self.loc_transform = loc_transform
        #aug transforms
        self.aug_transforms = [
            #vertical flipping
            torchvision.transforms.v2.RandomVerticalFlip(p = 1),
            #horizontal flipping
            torchvision.transforms.v2.RandomHorizontalFlip(p = 1),
            #random rotation
            torchvision.transforms.v2.RandomRotation((360), fill = 0),
            #cutout
            #Cutout()
            #mirroir shifts
            #Mirroir(img_h = height, img_w = width, shift_h = shift_h, shift_w = shift_w),
            #patchify
            #Patchify(img_h = height, img_w = width, patch_h = patch_h, patch_w = patch_w)
        ]
        #global augmentations
        self.glb_transforms = [
            #random scaling
            torchvision.transforms.v2.RandomAffine(degrees = 0, scale = (1.0, 1.2), fill = 0),
            #random perspective
            #torchvision.transforms.v2.RandomPerspective(distortion_scale = 0.2, p = 1, fill = -1),
        ]
        #local augmentations
        self.loc_transforms = [
            #random resized crop
            torchvision.transforms.v2.RandomResizedCrop((height, width), scale=(0.5, 0.9), ratio=(1.0, 1.0),
                                                        antialias = True),
        ]
    #call
    def __call__(self, img, type_ = 'glb'): #channelxheightxwidth
        #apply transform
        img = apply_transform(img, self.aug_p, self.aug_transform, self.aug_transforms)
        if type_ == 'glb':
            #apply transform
            img = apply_transform(img, self.glb_p, self.glb_transform, self.glb_transforms)
        elif type_ == 'loc':
            #apply transform
            img = apply_transform(img, self.loc_p, self.loc_transform, self.loc_transforms)
        #return
        return img

#mixed augmentations
class Mixer():
    #init
    def __init__(self, mix_p = 1.0, cutmix_alpha = 0.5, mixup_alpha = 0.5, num_classes = 2):
        self.mix_p = mix_p
        seeding()
        self.cutmix_or_mixup = torchvision.transforms.v2.RandomChoice([
            torchvision.transforms.v2.CutMix(alpha = cutmix_alpha, num_classes = num_classes), 
            torchvision.transforms.v2.MixUp(alpha = mixup_alpha, num_classes = num_classes)])
    #collate_fn    
    def collate_fn(self, batch):
        #collate
        images, labels, weights = default_collate(batch)
        #random
        if self.mix_p > random.random():
            #new
            new_images, new_labels = self.cutmix_or_mixup(images, labels)
            #return
            return new_images, new_labels, labels, weights
        else:
            #cloned
            new_labels = labels.clone()
            #return
            return images, new_labels, labels, weights

#validation collate_fn (hopefully, prevent noisy data)
def val_collate_fn(batch):
    #collate
    images, labels, weights = default_collate(batch)
    #apply transforms
    images_v = torchvision.transforms.v2.functional.vflip(images)
    #apply transforms
    images_h = torchvision.transforms.v2.functional.hflip(images)
    #apply transforms
    images_hv = torchvision.transforms.v2.functional.hflip(images_v)
    #concatenate
    images = torch.concatenate([images, images_v, images_h, images_hv], dim = 0) #x4
    #concatenate
    labels = torch.tile(labels, (images.shape[0] // labels.shape[0],)) 
    #concatenate
    weights = torch.tile(weights, (images.shape[0] // weights.shape[0],))
    #rotate 90
    images_90 = torchvision.transforms.v2.functional.rotate(images, angle = 90)
    #rotate 180
    images_180 = torchvision.transforms.v2.functional.rotate(images, angle = 180)
    #rotate 270 
    images_270 = torchvision.transforms.v2.functional.rotate(images, angle = 270)
    #concatenate
    images = torch.concatenate([images, images_90, images_180, images_270], dim = 0) #x4
    #concatenate
    labels = torch.tile(labels, (images.shape[0] // labels.shape[0],)) 
    #concatenate
    weights = torch.tile(weights, (images.shape[0] // weights.shape[0],))
    #return
    return images, labels, weights #x16

#custom image dataset
class CustomImageDataset(torch.utils.data.Dataset):
    #init
    def __init__(self, df, data = 'TRAIN', aug = False, label = 'Mod_Biomarker', 
                 col_mode = 'AF', input_size = (299, 299), norm_mean = (0.5, 0.5), norm_std = (0.5, 0.5), 
                 aug_p = 1.0, aug_transform = None, 
                 glb_p = 1.0, glb_transform = None, 
                 loc_p = 1.0, loc_transform = None):
        #dataframe
        self.df = df[df['DATA'] == data]
        #aug
        self.aug = aug
        #label
        self.label = label
        #col_mode
        self.col_mode = col_mode
        #input size
        self.input_size = input_size
        #resize
        self.resize = torchvision.transforms.v2.Resize(size = (
            self.input_size[0], self.input_size[1]), antialias = True)
        #norm
        self.norm = torchvision.transforms.v2.Normalize(norm_mean, norm_std)
        #seeding
        seeding()
        #check
        if self.aug:
            #transforms
            self.transforms = Custom_Augment(aug_p = aug_p, aug_transform = aug_transform, 
                                         glb_p = glb_p, glb_transform = glb_transform, 
                                         loc_p = loc_p, loc_transform = loc_transform)
    #len
    def __len__(self):
        return len(self.df)
    #get item
    def __getitem__(self, idx):
        #row
        row = self.df.iloc[idx]
        #image
        image = torch.Tensor(sitk.GetArrayFromImage(
            sitk.ReadImage(row['InputFilePath'])))
        #move axis
        image = torch.moveaxis(image, 2, 0)
        #norm
        image = self.resize(image)
        
        #label
        label = torch.tensor(row[self.label].item(), dtype = torch.long)
        #weight loss squeeze
        weights = torch.Tensor([row['Loss_Weight'].item()]).squeeze()
        #if augmentation
        if self.aug:
            #local transform
            image = self.transforms(image, type_ = None)
            #glb transform
            image = self.transforms(image, type_ = 'glb')
            #loc transform
            image = self.transforms(image, type_ = 'loc')
        #norm
        image = self.norm(image)
        #return
        return image, label, weights

#seed worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#get the optimizer
def get_optimizer(model_config, model):
    #AdamW
    if model_config['optimizer']['description'] == 'AdamW':
        #optimize parameter
        optimizer = torch.optim.AdamW(model.parameters(), lr = model_config['init_lr'],
                                      weight_decay = model_config['weight_decay'])
    #llrd
    if model_config['optimizer']['description'] == 'LLRD':
        #embedding layer
        parameters = [model.vit_encoder.embeddings.parameters(), ]
        #parameters + split layers [0, 15, 30]
        parameters = parameters + split_layers(model, n_layers = model_config['optimizer']['n_layers'])
        #parameters + layernorm, pooler, classifier
        parameters = parameters + [model.vit_encoder.layernorm.parameters(), 
                                   #model.pooler.parameters(), 
                                   model.classifier.parameters()]
        #reverse
        parameters.reverse()
        #init
        optimizer_parameters = []
        #iterate
        for i, param in enumerate(parameters):
            #append
            optimizer_parameters.append(
                {'params': param, 
                 'lr': model_config['init_lr'] * model_config['optimizer']['decay_factor'] ** i})
        #optimizer
        optimizer = torch.optim.AdamW(optimizer_parameters, 
                                      #lr = model_config['init_lr'],
                                      weight_decay = model_config['weight_decay'])
    #llrd wd
    if model_config['optimizer']['description'] == 'LLRD_WD':
        #parameter group vars
        parameter_group_vars = {}
        #iterate by name and parameter
        for i, (name, param) in enumerate(model.named_parameters()):
            #ignore parameters that do not require grad
            if not param.requires_grad:
                continue
            #exempt specific named layers from weight decay
            if any(sub in name for sub in ['layernorm', 'bias', 'cls_token', 'position_embeddings']):
                weight_decay = 0.0
            else:
                weight_decay = model_config['weight_decay']
            #set
            if 'vit_encoder.embeddings' in name:
                lr = model_config['init_lr'] * model_config['optimizer']['decay_factor']**3
            elif 'vit_encoder.encoder.layer' in name:
                lr = model_config['init_lr'] * model_config['optimizer']['decay_factor']**2
            elif 'vit_encoder.layernorm' in name:
                lr = model_config['init_lr'] * model_config['optimizer']['decay_factor']**1
            elif 'classifier' in name:
                lr = model_config['init_lr'] * model_config['optimizer']['decay_factor']**0
            else:
                lr = model_config['init_lr'] * model_config['optimizer']['decay_factor']**0
                print('Missing Layer:', name)
            #append
            parameter_group_vars[i] = {'name': name, 'params': param, 'lr': lr, 
                                       'weight_decay': weight_decay}
        #parameter values
        optimizer = torch.optim.AdamW(list(parameter_group_vars.values()))     
    #wd
    if model_config['optimizer']['description'] == 'WD':
        #parameter group vars
        parameter_group_vars = {}
        #iterate by name and parameter
        for i, (name, param) in enumerate(model.named_parameters()):
            #ignore parameters that do not require grad
            if not param.requires_grad:
                continue
            #exempt specific named layers from weight decay
            if any(sub in name for sub in ['layernorm', 'bias', 'cls_token', 'position_embeddings']):
                weight_decay = 0.0
            else:
                weight_decay = model_config['weight_decay']
            #append
            parameter_group_vars[i] = {'name': name, 'params': param, 
                                       'lr': model_config['init_lr'], 
                                       'weight_decay': weight_decay}
        #parameter values
        optimizer = torch.optim.AdamW(list(parameter_group_vars.values()))   
    #return
    return optimizer

#get scheduler
def get_scheduler(model_config, optimizer):
    #plateau
    if model_config['scheduler']['description'] == 'plateau':
        #get the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode = model_config['scheduler']['mode'],
                                                               factor = model_config['scheduler']['factor'],
                                                               patience = model_config['scheduler']['patience'],
                                                               threshold = model_config['scheduler']['threshold'],
                                                               threshold_mode = model_config['scheduler']['threshold_mode'],
                                                               cooldown = model_config['scheduler']['cooldown'],
                                                               min_lr = model_config['scheduler']['min_lr'],
                                                               eps = model_config['scheduler']['eps'],)
    #cosine
    if model_config['scheduler']['description'] == 'cosine':
        #get the scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = model_config['epochs'],
            eta_min = model_config['init_lr'] * 0.1 ** model_config['eta_min'],)
    #cosine restart
    if model_config['scheduler']['description'] == 'cosine_restart':
        #get the scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0 = model_config['scheduler']['T_0'], T_mult = model_config['scheduler']['T_mult'], 
            eta_min = model_config['init_lr'] * 0.1 ** model_config['eta_min'])
    #cosine with warmup
    if model_config['scheduler']['description'] == 'cosine_warmup':
        #get the scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                    lr_lambda=lambda epoch: (epoch + 1) / model_config['scheduler']['warmup_epochs'])
        #get the scheduler
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = model_config['epochs'] - model_config['scheduler']['warmup_epochs'],
            eta_min = model_config['init_lr'] * 0.1 ** model_config['eta_min'],)
        #chain
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cos_scheduler], 
                                 milestones=[model_config['scheduler']['warmup_epochs'] - 1])
    #return
    return scheduler

#class earlystopping
class EarlyStopping:
    #early stop if validation does not improve for given patience
    def __init__(self, model_config, verbose = True, trace_func = print):
        #set up
        self.patience = model_config['patience']
        self.delta = model_config['delta']
        self.verbose = verbose
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.best_model = False
        self.early_stop = False
    #call
    def __call__(self, val_loss):
        #neg val loss
        score = -val_loss
        #init condition
        if self.best_score is None:
            self.best_score = score
        #count number of times model failed to meet the condition
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.best_model = False
            #identify when early stopping is required
            if self.counter >= self.patience:
                self.early_stop = True
        #if the model shows best score
        else:
            #get the score and counter
            self.best_score = score
            self.counter = 0
            self.best_model = True
        #return the interl
        return self.early_stop, self.best_model

# Mukhoti et. al. implementation at https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
def adaECE_error_mukhoti(confs, preds, labels, num_bins=15):
    def histedges_equalN(x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, num_bins + 1), np.arange(npt), np.sort(x))
        # y = numpy.interp(x, xp, fp) returns the interpolated function values for pints in x using xp-fp points already given. 
        # Let x = np.array([0.26, 0.53, 0.61, 0.75, 0.94, 0.99])
        # np.linspace(0, 6, 3 + 1) -> array([ 0.,  2.,  4., 6.])
        # np.arange(6) -> array([0, 1, 2, 3, 4, 5])
        # np.interp(np.linspace(0, 6, 3+1), np.arange(6), np.sort(x)) -> array([0.26, 0.61, 0.94, 0.99])

    confidences, predictions, labels = torch.FloatTensor(confs), torch.FloatTensor(preds), torch.FloatTensor(labels)
    accuracies = predictions.eq(labels)
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach()))

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1)
    bin_dict = collections.defaultdict(dict)
    bin_num = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_in_bin = torch.zeros(1)
            avg_confidence_in_bin = torch.zeros(1)
        # Save the bin stats to be returned
        bin_dict[bin_num]['lower_bound'] = bin_lower
        bin_dict[bin_num]['upper_bound'] = bin_upper
        bin_dict[bin_num]['prop_in_bin'] = prop_in_bin.item()
        bin_dict[bin_num]['accuracy_in_bin'] = accuracy_in_bin.item()
        bin_dict[bin_num]['avg_confidence_in_bin'] = avg_confidence_in_bin.item()
        bin_dict[bin_num]['calibration_gap'] = bin_dict[bin_num]['avg_confidence_in_bin'] - bin_dict[bin_num]['accuracy_in_bin']
        bin_num += 1
        
    return ece, bin_dict

#https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html
#ada focal loss
class AdaFocal(torch.nn.Module):
    def __init__(self, num_bins = 15, adafocal_lambda = 1.0, adafocal_gamma_initial = 1.0,
                 adafocal_switch_pt = 0.2, adafocal_gamma_max = 20.0,  adafocal_gamma_min = -2.0,
                 update_gamma_every = -1, save_folder = '', device = None):
        super(AdaFocal, self).__init__()
        self.num_bins = num_bins
        self.lamda = adafocal_lambda
        self.gamma_initial = adafocal_gamma_initial
        self.switch_pt = adafocal_switch_pt
        self.gamma_max = adafocal_gamma_max
        self.gamma_min = adafocal_gamma_min
        self.update_gamma_every = update_gamma_every
        self.save_folder = save_folder
        self.device = device
        # This initializes the bin_stats variable
        self.bin_stats = collections.defaultdict(dict)
        for bin_no in range(self.num_bins):
            self.bin_stats[bin_no]['lower_boundary'] = bin_no*(1/self.num_bins)
            self.bin_stats[bin_no]['upper_boundary'] = (bin_no+1)*(1/self.num_bins)
            self.bin_stats[bin_no]['gamma'] = self.gamma_initial

    # This function updates the bin statistics which are used by the Adafocal loss at every epoch.
    def update_bin_stats(self, val_adabin_dict):
        for bin_no in range(self.num_bins):
            # This is the Adafocal gamma update rule
            prev_gamma = self.bin_stats[bin_no]['gamma']
            exp_term = val_adabin_dict[bin_no]['calibration_gap']
            if prev_gamma > 0:
                next_gamma = prev_gamma * math.exp(self.lamda*exp_term)
            else:
                next_gamma = prev_gamma * math.exp(-self.lamda*exp_term)
            # This switches between focal and inverse-focal loss when required.
            if abs(next_gamma) < self.switch_pt:
                if next_gamma > 0:
                    next_gamma = -self.switch_pt
                else:
                    next_gamma = self.switch_pt
            self.bin_stats[bin_no]['gamma'] = max(min(next_gamma, self.gamma_max), self.gamma_min) # gamma-clipping
            self.bin_stats[bin_no]['lower_boundary'] = val_adabin_dict[bin_no]['lower_bound']
            self.bin_stats[bin_no]['upper_boundary'] = val_adabin_dict[bin_no]['upper_bound']
        # This saves the "bin_stats" to a text file.
        save_file = os.path.join(self.save_folder, "val_bin_stats.txt")
        with open(save_file, "a") as write_file:
            json.dump(self.bin_stats, write_file)
            write_file.write("\n")
        return

    # This function selects the gammas for each sample based on which bin it falls into.
    def get_gamma_per_sample(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            for bin_no, stats in self.bin_stats.items():
                if bin_no==0 and pt_sample < stats['upper_boundary']:
                    break
                elif bin_no==self.num_bins-1 and pt_sample >= stats['lower_boundary']:
                    break
                elif pt_sample >= stats['lower_boundary'] and pt_sample < stats['upper_boundary']:
                    break
            gamma_list.append(stats['gamma'])
        return torch.tensor(gamma_list).to(self.device)

    # This computes the loss value to be returned for back-propagation.
    def forward(self, input, target, weights = None):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        gamma = self.get_gamma_per_sample(pt)
        gamma_sign = torch.sign(gamma)
        gamma_mag = torch.abs(gamma)
        pt = gamma_sign * pt
        if weights is None:
            loss = -1 * ((1 - pt + 1e-20)**gamma_mag) * logpt # 1e-20 added for numerical stability
        else:
            loss = -1 * ((1 - pt + 1e-20)**gamma_mag) * logpt * weights.to(input.device) # 1e-20 added for numerical stability
        return loss.sum()

#crossentropyloss
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, label_smoothing = 0.0):
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
    def forward(self, y_pred, y_true, weights = None):
        #if isinstance(weights, torch.Tensor):
        if weights is None:
            loss = torch.nn.functional.cross_entropy(y_pred, y_true, 
                                                     label_smoothing = self.label_smoothing)
        else:
            #per sample loss
            per_sample_loss = torch.nn.functional.cross_entropy(y_pred, y_true, 
                                                                reduction = 'none', 
                                                                label_smoothing = self.label_smoothing)
            #multiply
            weighted_per_sample_loss = per_sample_loss * weights.to(y_pred.device)
            #mean
            loss = weighted_per_sample_loss.mean()
        #return
        return loss

#get criteria
def get_criterion(model_config):
    #adafocalloss
    if model_config['criterion']['description'] == 'adafocal':
        criterion = AdaFocal(num_bins = model_config['criterion']['num_bins'], 
                             adafocal_lambda = model_config['criterion']['adafocal_lambda'], 
                             adafocal_gamma_initial = model_config['criterion']['adafocal_gamma_initial'],
                             adafocal_switch_pt = model_config['criterion']['adafocal_switch_pt'], 
                             adafocal_gamma_max = model_config['criterion']['adafocal_gamma_max'],  
                             adafocal_gamma_min = model_config['criterion']['adafocal_gamma_min'],
                             update_gamma_every = -1, save_folder = model_config['save_folder'], device = device)
    #crossentropyloss (weighted)
    if model_config['criterion']['description'] == 'crossentropy':
        criterion = CrossEntropyLoss(label_smoothing = model_config['criterion']['label_smoothing'])
    #return
    return criterion

#format metrics
def format_metrics_to_gb(item, g_gigabyte = 2 ** 30):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num

#model saving policy
def save_model(model_config, model):
    #save model use weights instead
    model.eval()
    #depending on choice
    if model_config['save_weights_only']:
        torch.save(model.state_dict(), model_config['save_folder'] + 'model_weights.pth')
    else:
        torch.save(model, model_config['save_folder'] + 'model.pth')
    #save info
    save_params(model_config, model_config['save_folder'] + 'model_config.json')

#update previously saved config only
def update_config_stopearly(save_path):
    #load
    model_config = load_params(save_path + 'model_config.json')
    #update
    model_config['stopped_early'] = True
    #save
    save_params(model_config, save_path + 'model_config.json')

#new saving policy
def new_saving_policy(early_stop, best_model, model_config, model, epoch):
    #if there is an early stop
    if early_stop:
        #exit training
        exit_training = True
        #has the model been already saved?
        if model_config['model_criteria']:
            #save just the config with update
            update_config_stopearly(model_config['save_folder'])
        #if not already saved
        else:
            #update
            model_config['stopped_early'] = True
            #save the model and config
            save_model(model_config, model)
    #if there is not early stop
    else:
        #exit
        exit_training = False
        #need to log that we did not exit training early
        model_config['stopped_early'] = False
        #you want to save the model every n_epochs often
        if model_config['save_best_model'] == False:
          #check if epocch is divisible and nonzero
          if (epoch % model_config['save_after_n_epochs'] == 0) and (epoch != 0):
            #indicate the model was saved
            model_config['model_criteria'] = True
            #save
            save_model(model_config, model)
        else:
          #you want to save the best model
          if model_config['epochs_trained'] >= model_config['save_after_n_epochs']:
            #check if current model is the best model
            if best_model:
                #log that it is the best model
                model_config['best_model'] = True
                #indicate the model was saved
                model_config['model_criteria'] = True
                #then save
                save_model(model_config, model)
            #if current model is not the best model but want to save for the initital run
            if (best_model == False) and (model_config['epochs_trained'] == model_config['save_after_n_epochs']):
                #inidicate the model was saved
                model_config['model_criteria'] = True
                #save the model and config
                save_model(model_config, model)
    #return
    return model_config, exit_training

#class weights
def app_class_weights(df, model_config, device):
    #class counts from training set only
    class_counts = dict(df[df['Fold'] == 'TRAIN'][model_config['label']].value_counts())
    #calculate inverse frequency
    inverse_frequencies = {cls: 1.0 / count for cls, count in class_counts.items()}
    #calculate normalization factor
    sum_inverse_frequencies = sum(inverse_frequencies.values())
    #normalize weights
    normalized_weights = {cls: freq / sum_inverse_frequencies for cls, freq in inverse_frequencies.items()}
    #ensure the order of weights matches the class indices expected by your model/loss
    num_classes = max(class_counts.keys()) + 1
    class_weights = torch.zeros(num_classes)
    #iterate
    for cls, weight in normalized_weights.items():
        class_weights[cls] = weight
    #send to device
    return class_weights, class_weights.detach().cpu().numpy().tolist()
    
#freeze layers
def freeze_weights(net):
    for param in net.parameters():
        param.requires_grad = False

#init_weights
def weights_init(net):
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            torch.nn.init.zeros_(module.bias.data)

#load pretrained model
def load_pretrained_model(save_folder, device):
    #load
    model_config = load_params(save_folder + 'model_config.json')
    #weights
    if model_config['save_weights_only']:
        model = get_model(model_config)
        #important to do prior
        model.eval()
        #load weights
        model.load_state_dict(torch.load(model_config['save_folder'] + 'model_weights.pth', 
                                         map_location = device, weights_only = True))
    #send
    model.eval()
    model.to(device)
    #return
    return model, model_config

#self attention pooling https://arxiv.org/pdf/2008.01077
class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        #a single linear layer to compute attention scores for each element.
        self.attention_weights_layer = torch.nn.Linear(in_dim, 1)
    #forward
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len, in_dim).
            mask (torch.Tensor, optional): A boolean mask tensor of shape (batch_size, seq_len),
                where `True` indicates a valid element and `False` indicates a padded one.
                If `None`, no masking is applied.
        Returns:
            torch.Tensor: The pooled output tensor of shape (batch_size, in_dim).
        """
        #calculate attention scores for each element in the sequence.
        #scores shape: (batch_size, seq_len, 1)
        attention_scores = self.attention_weights_layer(x)
        #squeeze the last dimension to get shape (batch_size, seq_len).
        attention_scores = torch.squeeze(attention_scores, dim=2)
        #apply mask by filling padded scores with a very large negative value.
        #this makes their weight nearly zero after the softmax operation.
        if mask is not None:
            #the `masked_fill` function replaces values where the condition is `True`.
            #we want to find the padded elements, so we use `~mask`.
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        #apply softmax to normalize the scores into attention weights.
        #weights shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        #reshape weights for element-wise multiplication.
        #reshaped_weights shape: (batch_size, seq_len, 1)
        reshaped_weights = attention_weights.unsqueeze(2)
        #calculate the weighted sum of the input sequence.
        #pooled_output shape: (batch_size, in_dim)
        pooled_output = torch.sum(x * reshaped_weights, dim=1)
        #return
        return pooled_output

#max pooling
class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
    #forward
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        embeddings: (batch_size, seq_len, in_dim)
        mask: (batch_size, seq_len) - True for valid, False for padding
        """
        if mask is not None:
            # expand mask to (batch_size, seq_len, in_dim) to match embeddings
            mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
            # eeplace padding with -inf so they are ignored by max pooling
            # use .bool() and logical NOT (~) to target padding (False/0 values)
            embeddings = embeddings.masked_fill(~mask_expanded.bool(), -float('inf'))
        # global max pooling across the sequence dimension (dim=1)
        pooled_embeddings, _ = torch.max(embeddings, dim=1)
        # return
        return pooled_embeddings

#average pooling
class AveragePooling(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    #forward
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        embeddings (batch_size, seq_len, in_dim).
        mask (batch_size, seq_len) - True for valid, False for padding
        """
        if mask is not None:
            #expand mask  to match embeddings
            mask_expanded = mask.unsqueeze(-1).to(embeddings.dtype)
            #multiplying by mask zeros out padding values
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim = 1)
            #clamp ensures no division by zero
            token_counts = torch.clamp(mask_expanded.sum(dim = 1), min = 1e-9)
            #return true mean of valid tokens
            return sum_embeddings / token_counts
        else:
            #average pool
            return torch.mean(embeddings, dim = 1)

#expert nn
class Expert_NN(torch.nn.Module):
    #init
    def __init__(self, in_feature, dropout_p, num_classes = 2):
        super().__init__()
        #self.in_feature = in_feature
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_feature, in_feature // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(p = dropout_p),
            torch.nn.Linear(in_feature // 4, in_feature // 16),
            torch.nn.GELU(),
            torch.nn.Dropout(p = dropout_p),
            torch.nn.Linear(in_feature // 16, num_classes))
    #forward
    def forward(self, x):
        return self.layers(x)

# copy mae2vit (for the forward pass)
def mae2vit(model, model_config):
    #vit model get base encoder
    base_encoder = transformers.ViTModel.from_pretrained(
        model_config['pretrained_path'],
        num_channels = model_config['in_channels'],
        attn_implementation = 'eager',
        add_pooling_layer = False,
        force_download = True, 
        from_tf = False, 
        ignore_mismatched_sizes = True,
        #config = config)
    )
    #set source state dict
    source_state_dict = model.model.vit.state_dict()
    #match keys
    base_encoder.load_state_dict(source_state_dict, strict = True)
    #return
    return base_encoder

#vision transformer classifier
class ViTClassifier(torch.nn.Module):
    #init
    def __init__(self, vit_folder, num_classes = 2, 
                 cls_token = True, dropout_p = 0.1, pooler = 'max', 
                 classifier = 'expert', freeze_encoder = True, 
                 encoder_embedding_dropout = 0.1, 
                 encoder_attention_dropout = 0.2, 
                 encoder_output_dropout = 0.3, reload_weights = True):
        #super
        super(ViTClassifier, self).__init__()
        #load the vit (encoder + decoder)
        model, model_config = load_pretrained_model(save_folder = vit_folder, device = 'cpu')
        #cls_token
        self.cls_token = cls_token
        #interpolate
        self.interp = model.interpolate_pos_encoding
        #vit encoder
        if reload_weights:
            #vit encoder
            self.vit_encoder = mae2vit(model, model_config)
        else:
            #vit encoder
            self.vit_encoder = model.model.vit
        #encoder dropout
        set_encoder_dropout(self.vit_encoder, 
                            encoder_embedding_dropout = encoder_embedding_dropout,
                            encoder_attention_dropout = encoder_attention_dropout, 
                            encoder_output_dropout = encoder_output_dropout)
        #none
        model = None
        #freeze vit encoder
        if freeze_encoder:
            self.vit_encoder.apply(freeze_weights)
        #pooler
        if pooler == 'atn':
            self.pooler = SelfAttentionPooling(in_dim = self.vit_encoder.config.hidden_size)
        elif pooler == 'max':
            self.pooler = MaxPooling()
        elif pooler == 'avg':
            self.pooler = AveragePooling()
        else:
            self.pooler = None
        #classifier
        if classifier == 'expert':
            #expert nn
            self.classifier = Expert_NN(self.vit_encoder.config.hidden_size, 
                                        dropout_p, num_classes)
        else:
            #classifier
            self.classifier = torch.nn.Linear(self.vit_encoder.config.hidden_size, num_classes)
        #init weights
        self.classifier.apply(weights_init)
    #forward
    def forward(self, x):
        #embedding (cls token)
        embed = self.vit_encoder(x.to(device), 
                                 interpolate_pos_encoding = self.interp).last_hidden_state
        #cls_token
        if self.cls_token:
            logits = self.classifier(embed[:, 0, :])
        else:
            #pooling 
            logits = self.classifier(self.pooler(embed))
        #return
        return logits

#split layers
def split_layers(model, n_layers = [0, 10, 20, 30]):
    #ls
    ls = []
    #iterate
    for i in range(len(n_layers) - 1):
        #append
        ls.append(model.vit_encoder.encoder.layer[n_layers[i]:n_layers[i+1]].parameters())
    #append
    ls.append(model.vit_encoder.encoder.layer[n_layers[i+1]:].parameters())
    #return
    return ls

#kfold
def kfold(df, k_fold, model_config):
    #if
    if type(k_fold) is list:
        print('Shifted Test Set')
        #set
        val_kfold = k_fold[0]
        test_kfold = k_fold[1]
        #validation data
        df_val = df[df['Fold'] == val_kfold]
        df_val['DATA'] = 'VAL'
        #testing data
        df_test = df[df['Fold'] == test_kfold]
        df_test['DATA'] = 'TEST'
        #training
        df_train = df[(df['Fold'] != val_kfold) & (df['Fold'] != test_kfold)]
        df_train['DATA'] = 'TRAIN'
        #concat
        df_data = pd.concat([df_train, df_val, df_test]).reset_index(drop = True)
        #data flag
        data_flag = 'VAL'
    elif type(k_fold) is int:
        print('Out of Sample Test Set')
        #test
        df_test = df[df['Fold'] == 'TEST']
        df_test['DATA'] = 'TEST'
        #validation
        df_val = df[df['Fold'] == k_fold]
        df_val['DATA'] = 'VAL'
        #training not validation or test data
        df_train = df[(df['Fold'] != k_fold) & (df['Fold'] != 'TEST')]
        df_train['DATA'] = 'TRAIN'
        #concat
        df_data = pd.concat([df_train, df_val, df_test]).reset_index(drop = True)
        #data flag
        data_flag = 'VAL' #can be VAL
    #return the data and the test flag
    df_data.to_pickle(model_config['save_folder'] + 'df_results.pkl')
    #return the data and test flag
    return df_data, data_flag

#get classifier
def get_classifier(model_config):
    seeding()
    return timm.create_model(model_config['architecture'],
                             pretrained = True,
                             in_chans = model_config['in_chans'],
                             num_classes = model_config['num_classes'])

#get val
def get_val(df, val = True):
    #remove any with test
    df = df[df['Fold'] != 'TEST']
    #if val is True // use shifted cross validation
    if val:
        min_ = df['Fold'].unique().tolist() + [int(df['Fold'].min())]
        ls = create_shifted_lists(min_, sublist_size = 2, step_size = 1)
    else:
        ls = df['Fold'].unique().tolist()
    #return
    return ls

#create shifted lists
def create_shifted_lists(data_list, sublist_size, step_size):
    """
    Creates a list of lists containing elements shifted by 'step_size'.

    Args:
        data_list (list): The original list.
        sublist_size (int): The number of elements in each sublist.
        step_size (int): The number of elements to shift for the next sublist.

    Returns:
        list: A list of the generated sublists.
    """
    shifted_lists = []
    for i in range(0, len(data_list), step_size):
        # Slice the list to get a sublist of the specified size
        sublist = data_list[i:i + sublist_size]
        
        # Optional: Ensure sublists are the exact specified size
        # If the last sublist is shorter, you can choose to discard it or pad it
        if len(sublist) == sublist_size:
            shifted_lists.append(sublist)
        
    return shifted_lists

#for the loss function
def get_loss_weights(df, col = 'Cat_Stratify'):
    #class counts
    class_counts = dict(df[col].value_counts())
    #init
    class_weights = {}
    #iterate
    for cls, count in class_counts.items():
        class_weights[cls] = len(df) / (len(class_counts) * count)
    #apply
    df['Loss_Weight'] = df[col].map(class_weights)
    return df

#for the sampler
def get_sampler_weights(df, col = 'Cat_Stratify'):
    #class counts
    class_counts = dict(df[col].value_counts())
    #init
    class_weights = {}
    #iterate
    for cls, count in class_counts.items():
        class_weights[cls] = 1 / (count)
    #apply
    df['Sampler_Weight'] = df[col].map(class_weights)
    return df

#cls
def get_cls(row):
    if row['Class_Label'] == row['Class_Prediction'] == 1:
        label = 'TP'
    elif row['Class_Label'] == row['Class_Prediction'] == 0:
        label = 'TN'
    elif (row['Class_Label'] == 1) and (row['Class_Prediction'] == 0):
        label = 'FN'
    elif (row['Class_Label'] == 0) and (row['Class_Prediction'] == 1):
        label = 'FP'
    return label

#get_batch_metric
def get_batch_metric(y_true, y_pred):
    #small
    df_small = pd.DataFrame()
    #batch_cls
    batch_cls = y_pred.detach().cpu()
    #obtain the individual class predictions
    pred_prob = torch.softmax(batch_cls, dim = 1)
    #obtain the class prediction by probability (confidence value, predicted class)
    pred_conf, pred_class = torch.max(pred_prob, dim = 1)
    #class model prediction
    df_small['Class_Prediction'] = pred_class
    #confidence scores
    df_small['Class_Confidence'] = pred_conf
    #individual class probabilities
    for j in range(batch_cls.shape[1]):
        pred_tag = 'Class_' + str(j) + '_Prob'
        df_small[pred_tag] = pred_prob[:, j]
    #class label
    df_small['Class_Label'] = y_true
    #get_cls
    df_small['CLS'] = df_small.apply(get_cls, axis = 1)
    #return 
    return df_small

#compute_metrics (1)
def compute_metrics(tp, tn, fp, fn):
    try:
        #calculate precision
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision = 0 #None #np.nan
    try:
        #calculate negative predictive value
        neg_pred_value = tn/(tn+fn)
    except ZeroDivisionError:
        neg_pred_value = 0 #None #np.nan
    try:
        #get the recall or sensitivity (binary)
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall = 0 #None #np.nan
    #get the specificity
    try:
        specificity = tn/(tn+fp)
    except ZeroDivisionError:
        specificity = 0 #None #np.nan
    try:
        #get the f1 score
        f1 = (2*precision*recall)/(precision + recall)
    except:
        f1 = 0 #None #np.nan
    try:
        #get the accuracy
        accuracy = (tn+tp)/(tn+fp+fn+tp)
    except:
        accuracy = 0 #None #np.nan
    #return
    return precision, neg_pred_value, recall, specificity, f1, accuracy

#get_metrics
def get_metrics(df):
    #count true positive
    tp = df['CLS'].value_counts().get('TP', 0)
    #true negative
    tn = df['CLS'].value_counts().get('TN', 0)
    #false positive
    fp = df['CLS'].value_counts().get('FP', 0)
    #false negative
    fn = df['CLS'].value_counts().get('FN', 0)
    #binary classification (1)
    prc_1, npv_1, rec_1, spc_1, f1s_1, acc_1 = compute_metrics(tp, tn, fp, fn)
    #average precision score (1)
    aps_1 = sklearn.metrics.average_precision_score(df['Class_Label'], df['Class_1_Prob'])
    #roc auc score (1)
    auc_1 = sklearn.metrics.roc_auc_score(df['Class_Label'], df['Class_1_Prob'])
    #get objectives
    aps_auc_1, aps_auc_loss_1 = objective(aps_1, auc_1)
    #return
    return prc_1, npv_1, rec_1, spc_1, f1s_1, acc_1, aps_1, auc_1, aps_auc_1, aps_auc_loss_1

#pr-auc and roc-auc harmonic mean (f1)
def objective(pr_auc, roc_auc):
    metric = (2 * pr_auc * roc_auc) / (pr_auc + roc_auc)
    return metric, 1 - metric

#select_param()
def select_param(dict_config, trial):
    #iterate
    for key in dict_config.keys():
        #dict
        if type(dict_config[key]) is dict:
            #if categorical
            if 'categorical' in dict_config[key].keys():
                #update
                dict_config[key] = trial.suggest_categorical(key, dict_config[key]['categorical'])
    #return
    return dict_config

#get_delta
def get_delta(df, label = 'Label', interval = 10):
    #return
    return df.reset_index(drop = True)

#get category
def get_cat(df, 
            cols = [
                'description']):
    #stratify
    df['Cat_Stratify'] = df[cols].astype(str).agg('-'.join, axis = 1)
    #return
    return df

#apply splitting
def app_split(df, label, num_folds, random_state = None, 
              group = 'Group', rs_min = 0, rs_max = 1000):
    #optimal splitting
    if random_state is None:
        #random state
        df, df_rs, rs = opt_split(df, folds = num_folds, 
                                  group = group, stratify = 'Cat_Stratify', 
                                  label = label, 
                                  rs_min = rs_min, rs_max = rs_max)
    else:
        #stratify
        df = split_data(df, folds = num_folds, 
                        group = group, stratify = 'Cat_Stratify', 
                        rs = random_state)
        #set
        rs = random_state
    #return
    return df, rs

#optimum split 
def opt_split(df, folds = 5, group = 'Group', 
              stratify = 'Cat_Stratify', 
              label = 'Label', 
              rs_min = 0, rs_max = 100):
    #reset
    df = df.reset_index(drop = True)
    #warnings
    with warnings.catch_warnings():
        #ignore all caught warnings
        warnings.filterwarnings('ignore')
        #ls
        ls = []
        #iterate
        for rs in range(rs_min, rs_max):
          #get_folds
          df_ = split_data(df, folds = folds, group = group, stratify = stratify, rs = rs)
          #groupby
          x_ = df_.groupby(['Fold'])[label].mean().to_list()
          #compute the standard deviation
          std = statistics.stdev(x_)
          #append
          ls.append([rs, std])
        #get random state
        df_rs = pd.DataFrame(ls, columns = ['RS', 'SD']).sort_values(['SD'])
        #select the final and view
        rs = df_rs['RS'].iloc[0]
        #get folds and random state
        df_ = split_data(df, folds = folds, group = group, stratify = stratify, rs = rs)
    #return 
    return df_, df_rs, int(rs)

#split data
def split_data(df, folds = 5, group = 'Group', stratify = 'Label', rs = 42):
    #reset
    df = df.reset_index(drop = True)
    #the splitter
    skf = sklearn.model_selection.StratifiedGroupKFold(n_splits = folds,
                                                       shuffle = True,
                                                       random_state = rs)
    #set the target
    target = df.loc[:, stratify].astype('category')
    #set the group
    group_ids = df[group]
    #df_folds
    df_folds = []
    #iterate
    for i, (_, fold_index) in enumerate(skf.split(df, target, group_ids)):
        #set fold
        df_fold = df.loc[fold_index,:]
        #set
        df_fold['Fold'] = i
        #append
        df_folds.append(df_fold)
    #concat
    df = pd.concat(df_folds).reset_index(drop = True)
    #return
    return df

#def sweep objective
def sweep_objective(trial, sweep_config):
    #copy
    dict_config = sweep_config.copy()
    #select parameters
    dict_config = select_param(dict_config, trial)
    #read the pickle file with the dataset
    df_dataset = pd.read_pickle(dict_config['save_folder'] + 'df_sweep.pkl')
    #group name
    dict_config['group'] = f'trial-{trial.number}-{trial.datetime_start.strftime("%Y-%m-%d-%H-%M-%S")}'
    #get validation policy
    dict_config['val_test_folds'] = get_val(df_dataset, dict_config['val'])
    #group config copy
    group_config = dict_config.copy()
    #group
    ls_group = []
    ls_group_train_aps = []
    ls_group_train_auc = []
    ls_group_train_aps_auc = []
    ls_group_val_aps = []
    ls_group_val_auc = []
    ls_group_val_aps_auc = []
    #iterate by test_kfold
    for j, val_test_fold in enumerate(group_config['val_test_folds']):
        #train_aps = 0
        #train_auc = 0
        #train_aps_auc = 0
        #val_aps = 0
        #val_auc = 0
        #val_aps_auc = 0
        #cudnn for reproducibility
        #cudnn_(cudnn = False, deterministic = True, benchmark = False)
        #seeding for reproducibility
        #seeding()
        #copy
        model_config = dict_config.copy()
        #name the model
        model_config['model'] = model_config['group'] + '-' + model_config['mode'
            ].lower() + '-classifier-' + str(j) +  datetime.now().strftime('-%Y-%m-%d-%H-%M-%S')
        #save location
        model_config['save_folder'] = model_config['save_folder'] + model_config['model'] + '/'
        #create the model folder
        if os.path.isdir(model_config['save_folder']) == False:
            os.mkdir(model_config['save_folder'])
        #save results file
        df, data_flag = kfold(df_dataset, val_test_fold, model_config)
        #print
        print(model_config['save_folder'])
        print(df['DATA'].value_counts())
        #set val
        model_config['val_fold'] = val_test_fold[0]
        #set test
        model_config['test_fold'] = val_test_fold[1]
        #get model
        model = get_classifier(model_config)
        #training custom image dataset
        dset_train = CustomImageDataset(
            df = df, data = 'TRAIN', 
            input_size = model_config['input_size'],
            norm_mean = model_config['norm_mean'], norm_std = model_config['norm_std'],
            label = model_config['label'], 
            col_mode = model_config['mode'],
            aug = model_config['aug'], 
            aug_p = model_config['aug_p'], aug_transform = model_config['aug_transform'], 
            glb_p = model_config['glb_p'], glb_transform = model_config['glb_transform'], 
            loc_p = model_config['loc_p'], loc_transform = model_config['loc_transform']
        )
        #testing custom image dataset
        dset_val = CustomImageDataset(
            df = df, data = 'VAL', 
            input_size = model_config['input_size'],
            norm_mean = model_config['norm_mean'], norm_std = model_config['norm_std'],
            label = model_config['label'], 
            col_mode = model_config['mode'],
            aug = False, 
            aug_p = model_config['aug_p'], aug_transform = model_config['aug_transform'], 
            glb_p = model_config['glb_p'], glb_transform = model_config['glb_transform'], 
            loc_p = model_config['loc_p'], loc_transform = model_config['loc_transform']
        )
        #seed worker generator
        generator = torch.Generator()
        #manual seed
        generator.manual_seed(42)
        #mixer
        #mixer = Mixer(mix_p = model_config['mix_p'], 
                      #cutmix_alpha = model_config['cutmix_alpha'], 
                      #mixup_alpha = model_config['mixup_alpha'])
        #if applying sampler
        if model_config['sampler']:
            #get sampler
            train_sampler = torch.utils.data.WeightedRandomSampler(
            weights = dset_train.df['Sampler_Weight'].to_list(), 
            num_samples = len(dset_train.df),
            replacement = True)
            #get sampler (may interfere with validation computation)
            #val_sampler = torch.utils.data.WeightedRandomSampler(
                #weights = dset_val.df['Sampler_Weight'].to_list(), 
                #num_samples = len(dset_val.df),
                #replacement = True)
            val_sampler = None
            shuffle = False
        else:
            train_sampler, val_sampler, shuffle = (None, None, False)
        #get the train loader
        train_loader = torch.utils.data.DataLoader(dataset = dset_train,
                                                   batch_size = model_config['train_batch_size'],
                                                   num_workers = cpu_count,
                                                   generator = generator,
                                                   worker_init_fn = seed_worker,
                                                   collate_fn = None, #mixer.collate_fn,
                                                   sampler = train_sampler,
                                                   shuffle = shuffle)
        #get the val loader
        val_loader = torch.utils.data.DataLoader(dataset = dset_val,
                                                 batch_size = model_config['val_batch_size'],
                                                 num_workers = cpu_count,
                                                 generator = generator,
                                                 worker_init_fn = seed_worker,
                                                 collate_fn = None, #val_collate_fn,
                                                 sampler = val_sampler,
                                                 shuffle = False)
        #to device
        model.to(device)
        #model = torch.nn.DataParallel(model).to(device)
        #optimizer
        optimizer = get_optimizer(model_config, model)
        #scheduler
        scheduler = get_scheduler(model_config, optimizer)
        #early stopping (save time during the sweep)
        early_stopper = EarlyStopping(model_config)
        #get criterion
        criterion = get_criterion(model_config)
        #init
        ls_lr = []
        ls_train_metrics = []
        ls_val_metrics = []
        #iterate through the entire dataset
        for epoch in range(model_config['epochs_trained'] + 1, model_config['epochs'] + 1):
            #list of df train
            ls_df_train = []
            #determine train losses
            train_epoch_loss = 0
            #set for training
            model.train()
            #iterate through the training data
            for i, (x_true, y_true, weights) in enumerate(train_loader):
                #zero optimizer
                optimizer.zero_grad()
                #model 
                y_pred = model(x_true.to(device))
                #compute loss
                if model_config['weight_loss']:
                    loss = criterion(y_pred, y_true.to(device), weights = weights)
                else:
                    loss = criterion(y_pred, y_true.to(device), weights = None)
                #backward
                loss.backward()
                #step
                optimizer.step()
                #add
                #train_epoch_loss += loss.item()
                #batch metric training
                #ls_df_train.append(get_batch_metric(y_true, y_pred))
                #clear
                x_true, y_true, weights, loss = (None, None, None, None)
                #reset
                if device == 'cuda':
                    torch.cuda.empty_cache()
                #break
                #if i == 1:
                    #break
            #eval
            model.eval()
            #iterate through the training data
            for i, (x_true, y_true, weights) in enumerate(train_loader):
                #torch no grad
                with torch.no_grad():
                    #model
                    y_pred = model(x_true.to(device))
                    #compute loss
                    if model_config['weight_loss']:
                        loss = criterion(y_pred, y_true.to(device), weights = weights)
                    else:
                        loss = criterion(y_pred, y_true.to(device), weights = None)
                #add
                train_epoch_loss += loss.item()
                #batch metric validation
                ls_df_train.append(get_batch_metric(y_true, y_pred))
                #clear
                x_true, y_true, weights, loss = (None, None, None, None)
                #reset
                if device == 'cuda':
                    torch.cuda.empty_cache()
                #break
                #if i == 1:
                    #break
            #average
            train_epoch_loss /= len(train_loader)
            #training
            df_train = pd.concat(ls_df_train).reset_index(drop = True)
            #save
            df_train.to_pickle(model_config['save_folder'] + 'df_train_' + str(epoch) + '.pkl')
            #get metrics
            train_prc, train_npv, train_rec, train_spc, train_f1s, train_acc, train_aps, train_auc, train_aps_auc, train_aps_auc_loss = get_metrics(df_train)
            #list of df val
            ls_df_val = []
            #determine val loss
            val_epoch_loss = 0
            #set for validation
            model.eval()
            #iterate through the validation data
            for i, (x_true, y_true, weights) in enumerate(val_loader):
                #torch no grad
                with torch.no_grad():
                    #model
                    y_pred = model(x_true.to(device))
                    #compute loss
                    if model_config['weight_loss']:
                        loss = criterion(y_pred, y_true.to(device), weights = weights)
                    else:
                        loss = criterion(y_pred, y_true.to(device), weights = None)
                #add
                val_epoch_loss += loss.item()
                #batch metric validation
                ls_df_val.append(get_batch_metric(y_true, y_pred))
                #clear
                x_true, y_true, weights, loss = (None, None, None, None)
                #reset
                if device == 'cuda':
                    torch.cuda.empty_cache()
                #break
                #if i == 1:
                    #break
            #average
            val_epoch_loss /= (len(val_loader))# * 16) #x16 --> from val_collate_fn
            #validation
            df_val = pd.concat(ls_df_val).reset_index(drop = True)
            #save
            df_val.to_pickle(model_config['save_folder'] + 'df_val_' + str(epoch) + '.pkl')
            #get metrics
            val_prc, val_npv, val_rec, val_spc, val_f1s, val_acc, val_aps, val_auc, val_aps_auc, val_aps_auc_loss = get_metrics(df_val)
            #print
            print('EPOCH {0} OF {1} WITH LR {2:.2g} TRAIN LOSS {3:.2g} VAL LOSS {4:.2g}'.format(
                epoch, model_config['epochs'], optimizer.param_groups[0]['lr'], train_epoch_loss, val_epoch_loss))
            print('TRAIN PRC {0:.2g} NPV {1:.2g} REC {2:.2g} SPC {3:.2g} F1 {4:.2g} ACC {5:.2g} APS {6:.2g} AUC {7:.2g} APS-AUC {8:.2g}'.format(
                train_prc, train_npv, train_rec, train_spc, train_f1s, 
                train_acc, train_aps, train_auc, train_aps_auc))
            print('VAL PRC {0:.2g} NPV {1:.2g} REC {2:.2g} SPC {3:.2g} F1 {4:.2g} ACC {5:.2g} APS {6:.2g} AUC {7:.2g} APS-AUC {8:.2g}'.format(
                val_prc, val_npv, val_rec, val_spc, val_f1s, 
                val_acc, val_aps, val_auc, val_aps_auc))
            #append the results
            ls_lr.append(optimizer.param_groups[0]['lr'])
            ls_train_metrics.append([train_epoch_loss, train_prc, train_npv, train_rec, train_spc, 
                                     train_f1s, train_acc, train_aps, train_auc, train_aps_auc])
            ls_val_metrics.append([val_epoch_loss, val_prc, val_npv, val_rec, val_spc, val_f1s, 
                                   val_acc, val_aps, val_auc, val_aps_auc])
            #keep track of each epoch
            model_config['epochs_trained'] = epoch
            #log results training
            model_config['log_train_metrics'] = ls_train_metrics
            #log results validation
            model_config['log_val_metrics'] = ls_val_metrics
            #log the init_lr
            model_config['log_lr'] = ls_lr
            #scheduler // can use validation loss
            if model_config['scheduler']['description'] == 'plateau':
                if data_flag == 'TEST':
                    scheduler.step(1 - train_aps)
                elif data_flag == 'VAL':
                    scheduler.step(1 - val_aps) #1 - val_aps
            if 'cosine' in model_config['scheduler']['description']:
                scheduler.step()
            #check if after save n epochs
            if epoch >= model_config['save_after_n_epochs']:
                #determine if early stopping is required by training loss // can use validation loss
                if data_flag == 'TEST':
                    early_stop, best_model = early_stopper(1 - train_aps)
                elif data_flag == 'VAL':
                    early_stop, best_model = early_stopper(1 - val_aps) #1 - val_aps
            else:
                early_stop, best_model = (False, False)
            #new saving policy
            model_config, exit_training = new_saving_policy(
                early_stop, best_model, model_config, model, epoch)
            #set if exiting training
            if exit_training:
                print('Early Stop: Exit Training')
                break
        #model
        model, optimizer, train_loader, val_loader = (None, None, None, None)
        scheduler, criterion, early_stopper = (None, None, None)
        #reset
        if device == 'cuda':
            torch.cuda.empty_cache()
        #append
        ls_group.append(model_config['save_folder'])
        #add
        #ls_group_train_aps.append(train_aps)
        #ls_group_train_auc.append(train_auc)
        #ls_group_train_aps_auc.append(train_aps_auc)
        #ls_group_val_aps.append(val_aps)
        #ls_group_val_auc.append(val_auc)
        #ls_group_val_aps_auc.append(val_aps_auc)
    for i, sweep_folder in enumerate(ls_group):
        #load in the params
        model_config = load_params(sweep_folder + '/' + 'model_config.json')
        #train aps
        ls_group_train_aps.append(model_config['log_train_metrics'][-1][-3])
        #train auc
        ls_group_train_auc.append(model_config['log_train_metrics'][-1][-2])
        #train aps auc
        ls_group_train_aps_auc.append(model_config['log_train_metrics'][-1][-1])
        #val aps
        ls_group_val_aps.append(model_config['log_val_metrics'][-1][-3])
        #val auc
        ls_group_val_auc.append(model_config['log_val_metrics'][-1][-2])
        #val aps auc
        ls_group_val_aps_auc.append(model_config['log_val_metrics'][-1][-1])
    #divide
    group_train_aps = statistics.mean(ls_group_train_aps)
    group_train_auc = statistics.mean(ls_group_train_auc)
    group_train_aps_auc = statistics.mean(ls_group_train_aps_auc)
    group_val_aps = statistics.mean(ls_group_val_aps)
    group_val_auc = statistics.mean(ls_group_val_auc)
    group_val_aps_auc = statistics.mean(ls_group_val_aps_auc)
    #training metrics average precision score
    group_config['train_aps'] = ls_group_train_aps
    #training metrics auc
    group_config['train_auc'] = ls_group_train_auc
    #training metrics aps-auc
    group_config['train_aps_auc'] = ls_group_train_aps_auc
    #validation metrics average precision score
    group_config['val_aps'] = ls_group_val_aps
    #validation metrics auc
    group_config['val_auc'] = ls_group_val_auc
    #validation metrics aps-auc
    group_config['val_aps_auc'] = ls_group_val_aps_auc
    #print
    print(
        'TRIAL {0} TRAIN APS {1:.2g} AUC {2:.2g} APS-AUC {3:.2g} & VAL APS {4:.2g} AUC {5:.2g} APS-AUC {6:.2g}'.format(
            trial.number, 
            group_train_aps, group_train_auc, group_train_aps_auc, 
            group_val_aps, group_val_auc, group_val_aps_auc))
    #save_params
    save_params(group_config, dict_config['save_folder'] + 'group_' + str(trial.number) + '_config.json')
    #determine data flag
    if data_flag == 'TEST':
        #print
        print('Tuning on Training Set')
        #return
        return group_train_aps, group_train_aps_auc, group_train_auc
    elif data_flag == 'VAL':
        #print
        print('Tuning on Validation Set')
        #return
        return group_val_aps, group_val_aps_auc, group_val_auc

#sweep_config
sweep_config = {
    #name of the sweep
    'name': None,
    #description
    'description': '',
    #project
    'project': '',
    #model
    'model': None,
    #count number of trials
    'count': ,
    #data_path
    'data_path': '',
    #label
    'label': '',
    #save_folder
    'save_folder': '',
    #mode
    'mode': '',
    #technical parameters
    'gpu': gpu,
    'num_gpu': gpu_count,
    'cpu': get_processor_name(),
    'cpu_cores': multiprocessing.cpu_count(),
    'conda': get_micromamba_version(),
    'python': get_python_version(),
    #saving model
    'stopped_early': None,
    'best_model': None,
    'model_criteria': False,
    'save_weights_only': True,
    'save_best_model': True,
    'save_after_n_epochs': 2,
    'epochs_trained': 0,
    #early stopping
    'patience': 5,
    'delta': 1e-3,
    #pretrained path
    'architecture': '',
    #in channels
    'in_chans': 2,
    #num classes
    'num_classes': 2,
    #input size
    'input_size': (224, 224),
    #norm mean
    'norm_mean': (0.5, 0.5, ),
    #norm std
    'norm_std': (0.5, 0.5, ),
    #label
    'label': '', 
    #interval
    'interval': 5,
    #num folds
    'num_folds': 4, 
    #validation
    'val': True, 
    #augmentation parameters
    'aug': {'categorical': [True]},
    'aug_p': {'categorical': [0.25]}, #augmentation probability
    'aug_transform': {'categorical': [None]}, #augmentation transforms None --> max
    'glb_p': {'categorical': [0.1]}, #global probability
    'glb_transform': {'categorical': [None]}, #augmentation transform None --> max
    'loc_p': {'categorical': [0.1]}, #local probability
    'loc_transform': {'categorical': [None]}, #augmentation transform max
    #'mix_p': {'categorical': [0.0, 0.5]},
    #'cutmix_alpha': {'categorical': [0.2, 0.5, 0.8]},
    #'mixup_alpha': {'categorical': [0.2, 0.5, 0.8]},
    #set training parameters
    'train_batch_size': {'categorical': [5]},
    'val_batch_size': {'categorical': [5]},
    'optimizer': {'categorical': [#{'description': 'LLRD_WD', 'decay_factor': 0.1}, 
                                  #{'description': 'LLRD_WD', 'decay_factor': 0.5},
                                  #{'description': 'LLRD_WD', 'decay_factor': 0.9},
                                  {'description': 'AdamW'},
                                  #{'description': 'WD', 'decay_factor': 0.1}
                                 ]},
    'init_lr': {'categorical': [1e-6]},
    'eta_min': {'categorical': [1, 2]},
    'weight_decay': {'categorical': [0.0]},
    'criterion': {'categorical': [{'description': 'crossentropy', 'label_smoothing': 0.0},]},
    'weight_loss': {'categorical': [False]},
    'sampler': {'categorical': [False]},
    'epochs': {'categorical': [15]},
    'scheduler': {'categorical': [
        {'description': 'cosine_warmup', 'warmup_epochs': 3},
        #{'description': 'plateau', 'mode': 'min', 'factor': 1e-1, 'patience': 1, 
         #'threshold': 1e-3, 'threshold_mode': 'abs', 'cooldown': 1, 'min_lr': 1e-8, 'eps': 1e-8}
    ]},}
#main
if __name__ == '__main__':
    #cudnn for reproducibility
    cudnn_(cudnn = False, deterministic = True, benchmark = False)
    #seeding for reproducibility
    seeding()
    #specify the sweep save location
    if sweep_config['name'] is None:
        sweep_config['name'] = datetime.now().strftime(
            sweep_config['mode'].lower() + '-classifier-sweep-%Y-%m-%d-%H-%M-%S')
    #set
    sweep_config['save_folder'] = sweep_config['save_folder'] + sweep_config['name'] + '/'
    #create the sweep folder
    if os.path.isdir(sweep_config['save_folder']) == False:
        os.mkdir(sweep_config['save_folder'])
    #read pickle
    df = pd.read_pickle(sweep_config['data_path'])
    #get delta
    df = get_delta(df, label = sweep_config['label'], interval = sweep_config['interval'])
    #get category
    df = get_cat(df)
    #if sweep_config['parameters']['weight_loss']['value']:
    df = get_loss_weights(df, col = 'Cat_Stratify')
    #get loss weights if desired
    #if sweep_config['parameters']['sampler']['value']:
    df = get_sampler_weights(df, col = 'Cat_Stratify')
    #applying split
    df, rs = app_split(df, label = sweep_config['label'], num_folds = sweep_config['num_folds'], 
                       random_state = None, group = '', rs_min = 0, rs_max = 100)
    #random state
    sweep_config['random_state'] = rs
    #save pickle
    df.to_pickle(sweep_config['save_folder'] + 'df_sweep.pkl')
    #save the sweep config in the sweep folder
    save_params(sweep_config, sweep_config['save_folder'] + 'sweep_config.json')
    #create study
    study = optuna.create_study(
        directions = ['maximize', 'maximize', 'maximize'],
        study_name = sweep_config['name'],
        sampler = optuna.samplers.TPESampler(seed = 42),
        #pruner = optuna.pruners.MedianPruner(), #prune unpromising trials
        #load_if_exists = True
    )
    #set metric names
    study.set_metric_names(['aps', 'aps_auc', 'auc'])
    #optimize study
    study.optimize(partial(sweep_objective, sweep_config = sweep_config), 
                   n_trials = sweep_config['count'], 
                   show_progress_bar = True, gc_after_trial = True)
    #load the sweep config
    sweep_config = load_params(sweep_config['save_folder'] + 'sweep_config.json')
    #save the trials
    study.trials_dataframe().to_pickle(sweep_config['save_folder'] + 'df_trial.pkl')
    #init
    pareto_front_trials = study.best_trials
    #get max
    best_metric_trial = max(pareto_front_trials, key = lambda t: t.values[0])
    #maximization of objectives
    print('Maximize Objectives: APS: {0:.2g} APS-AUC: {1:.2g} AUC: {2:.2g}'.format(
        best_metric_trial.values[0], best_metric_trial.values[1], best_metric_trial.values[2]))
    #get best metrics
    sweep_config['best_metrics'] = best_metric_trial.values
    #best trial
    sweep_config['best_trial'] = best_metric_trial.number
    #get best params
    sweep_config['best_config'] = best_metric_trial.params
    #importances
    sweep_config['importance'] = optuna.importance.get_param_importances(
        study, target=lambda t: t.values[0])
    #save params
    save_params(sweep_config, sweep_config['save_folder'] + 'sweep_config.json')
    #number of trials
    print('Best Trial {0}'.format(best_metric_trial.number))
