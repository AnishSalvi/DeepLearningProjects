# install file

#imports
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #":16:8"
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
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
import glob
import sys
import timm

#set cpu count
cpu_count = multiprocessing.cpu_count() - 1
print('CPU Count', cpu_count)
os.environ['TOKENIZERS_PARALLELISM'] = 'TRUE'

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

#get classifier
#def get_classifier(model_config):
#    return ViTClassifier(
#        vit_folder = model_config['vit_folder'], 
#        num_classes = model_config['num_classes'],
#        cls_token = model_config['cls_token'],
#        dropout_p = model_config['dropout_p'],
#        pooler = model_config['pooler'],
#        classifier = model_config['classifier'], 
#        freeze_encoder = model_config['freeze_encoder'], 
#        encoder_embedding_dropout = model_config['encoder_embedding_dropout'],
#        encoder_attention_dropout = model_config['encoder_attention_dropout'], 
#        encoder_output_dropout = model_config['encoder_output_dropout']
#    )
#get classifier
def get_classifier(model_config):
    seeding()
    return timm.create_model(model_config['architecture'],
                             pretrained = True,
                             in_chans = model_config['in_chans'],
                             num_classes = model_config['num_classes'])

#get the model
def get_model(model_config):
    seeding()
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

#freeze layers
def freeze_weights(net):
    for param in net.parameters():
        param.requires_grad = False

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
        #hidden = self.compute_hidden(outputs.last_hidden_state[:, 1:, :])
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
        return decoder_outputs.logits, rec_loss

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

#unfreeze positional embeddings
def unfreeze_pos(model):
    for param in model.parameters():
        param.requires_grad = True
    return model

#compare models
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

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

#load pretrained classifier
def load_pretrained_classifier(save_folder, device):
    #load
    model_config = load_params(save_folder + 'model_config.json')
    #weights
    if model_config['save_weights_only']:
        model = get_classifier(model_config)
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

#split
def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))

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

#custom image dataset
class CustomImageDataset(torch.utils.data.Dataset):
    #init
    def __init__(self, df, col_mode = 'AF', input_size = (299, 299), norm_mean = (0.5, ), norm_std = (0.5, )):
        #dataframe
        self.df = df
        #col mode
        self.col_mode = col_mode
        #input size
        self.input_size = input_size
        #resize
        self.resize = torchvision.transforms.v2.Resize(size = (
            self.input_size[0], self.input_size[1]), antialias = True)
        #norm
        self.norm = torchvision.transforms.v2.Normalize(norm_mean, norm_std)
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
        #norm
        image = self.norm(image)
        #return
        return image

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

#cls
def get_cls(row, col = 'Binary Class'):
    if row[col] == row['Class_Prediction'] == 1:
        label = 'TP'
    elif row[col] == row['Class_Prediction'] == 0:
        label = 'TN'
    elif (row[col] == 1) and (row['Class_Prediction'] == 0):
        label = 'FN'
    elif (row[col] == 0) and (row['Class_Prediction'] == 1):
        label = 'FP'
    return label

#specify the sweep folder
sweep_path = '/'
#specify the model tag
model_tag = 'trial-*'
#batch size
infer_batch_size = 50
#mask
#mask_ratio = 0.0 #None is masking policy used for training

#main
if __name__ == '__main__':
    #cudnn for reproducibility
    cudnn_(cudnn = False, deterministic = True, benchmark = False)
    #seeding for reproducibility
    seeding()
    #get all the models in the sweep
    sweep_folders = sorted(glob.glob(sweep_path + model_tag + '*'))
    #iterate
    for sweep_folder in sweep_folders:
        #load in the params
        model, model_config = load_pretrained_classifier(sweep_folder + '/', device)
        #load in the results file
        df = pd.read_pickle(model_config['save_folder'] + 'df_results.pkl') #may need to change for cross val
        #need to obtain the data
        ls_df = split_given_size(df, infer_batch_size) #num patients in batch XXX
        #eval
        model.eval()
        #prep
        model.to(device)
        #init
        save_dfs = []
        #load the data
        dset_infer = CustomImageDataset(df, input_size = model_config['input_size'], 
                                        norm_mean = model_config['norm_mean'], 
                                        norm_std = model_config['norm_std'],
                                        col_mode = model_config['mode'])
        #load
        infer_loader = torch.utils.data.DataLoader(dset_infer, 
                                                   batch_size = infer_batch_size, 
                                                   num_workers = cpu_count) #num patients in batch XXX
        #mask ratio
        #model.vit_encoder.config.mask_ratio = mask_ratio
        #iterate
        for i, x_true in enumerate(infer_loader):
            #no gradient
            with torch.no_grad():
                #compute y_pred
                y_pred = model(x_true.to(device))
                #batch_cls
                batch_cls = y_pred.detach().cpu()
                #set to None
                y_pred = None
                #reset
                if device == 'cuda':
                    torch.cuda.empty_cache()
            #get the small df
            df_small = ls_df[i]
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
            #get classification
            df_small['CLS'] = df_small.apply(get_cls, axis = 1, args = ('CLS_CPT', ))
            #append
            save_dfs.append(df_small)
            #print
            print('Finished Batch: {0} of {1}'.format (i+1, len(ls_df))) 
        #concat
        df_pred = pd.concat(save_dfs)
        #technical parameters
        #model_config['infer_mask_ratio'] = mask_ratio
        model_config['infer_gpu'] = gpu
        model_config['infer_num_gpu'] = gpu_count
        model_config['infer_cpu'] = get_processor_name()
        model_config['infer_cpu_cores'] = multiprocessing.cpu_count()
        model_config['infer_micromamba'] = get_micromamba_version()
        model_config['infer_python'] = get_python_version()
        #save params
        save_params(model_config, model_config['save_folder'] + 'model_config.json')
        #save
        df_pred.to_pickle(model_config['save_folder'] + 'df_results.pkl')
        #print
        print('Completed: ', model_config['save_folder'])
        #clear
        model = None
        #clear
        if device == 'cuda':
            torch.cuda.empty_cache() 
