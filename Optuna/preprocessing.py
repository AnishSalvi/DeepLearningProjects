#linux terminal
#gcloud auth login
#gcloud auth application-default login
#gsutil -m cp -r save_folder gs://bucket_name/

#imports
import pandas as pd
import pydicom
import numpy as np
import torch
import torchvision
from google.cloud import storage
from google.cloud import bigquery
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import multiprocessing
import skimage
import filecmp
import itertools
from itertools import compress
print('CPU Count', multiprocessing.cpu_count() - 1)
pd.options.mode.chained_assignment = None

#download blob
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

#define bounding box of binary
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

#crop
def crop_image(image):
    #cropping
    mask_image = np.where(image > 0, 1, 0)
    #get coords
    xmin, xmax, ymin, ymax = bbox2(mask_image)
    #subset region of interest
    return image[xmin:xmax, ymin:ymax,]

#process_image
def process_image(row):
    #metrics
    metrics = {}
    #determine folder for dcm
    #dcm_folder = dcm_folder
    #determine folder for fun
    img_folder = save_folder
    #determine the filename for dcm 
    dcm_fname = row['Mode'] + '_' + str(row['Image']) + '.dcm'
    #determine the filename for
    img_fname = row['Mode'] + '_' + str(row['Image']) + '.nii.gz'
    #try
    if os.path.isfile(img_folder + img_fname) == False:
        try:
            #check if the image already exists
            if os.path.exists(img_folder + img_fname) == False:
                #download the blob for dcm oct
                if os.path.exists(dcm_folder + dcm_fname) == False:
                    #download_blob
                    download_blob(bucket_name, row['FilePath'], dcm_folder + dcm_fname)
                #read in the image
                image = pydicom.dcmread(dcm_folder + dcm_fname, force = True).pixel_array
                #check colour
                if row['Mode'] == 'CLR':
                    #keep only first two channels
                    image = image[:, :, :2]
                #crop
                image = crop_image(image)
                #resize
                #image = skimage.transform.resize(image, resize)
                #check af
                #if row['Mode'] == 'AF':
                    #expand dims
                    #image = np.expand_dims(image, axis = 2)
                #move axis
                #image = np.moveaxis(image, 2, 0)
                #tensor
                #image = torch.Tensor(image)
                #check colour
                #if row['Mode'] == 'CLR':
                    #normalize
                    #image = clr_normalize(image)
                #elif row['Mode'] == 'AF':
                    #normalize
                    #image = af_normalize(image)
            #save file after finished (optos_img)
            sitk.WriteImage(sitk.GetImageFromArray(image), img_folder + img_fname)
            #remove file after finished (optos_dcm)
            if (os.path.exists(dcm_folder + dcm_fname)): # and (save_space == True):
                os.remove(dcm_folder + dcm_fname)
            #collect
            metrics['InputFilePath'] = img_folder + img_fname
        except:
            #set
            metrics['InputFilePath'] = np.nan
    else:
        #collect
        metrics['InputFilePath'] = img_folder + img_fname
    #return
    return metrics

#get size
def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    #print
    print(start_path)
    print('{0:.2g} GB'.format(total_size * 1e-9))

#check if the same
def check_files(file_pair):
    file0 = file_pair[0]
    file1 = file_pair[1]
    return filecmp.cmp(file0, file1,
                       shallow = False)

#drop
def drop_df(df, col = 'InputFilePath'):
    #drop duplicates
    df_ = df.drop_duplicates(['Group', 'Laterality', 'Date', 'UniqueImage'])
    #isin
    df_tmp = df[~df[col].isin(df_[col])]
    #empty
    if df_tmp.empty:
        return df_
    else:
        ls_files = df_tmp[col].to_list()
        if ls_files:
            with multiprocessing.Pool() as pool:
                _ = pool.map(remove_file, ls_files)
        return df_

#remove unwanted files
def remove_file(file):
    os.remove(file)

#identify the number of duplicate files in a folder
def get_dup(files, print_ = False):
    #files = sorted(glob.glob(folder + '/*.dcm'))
    # List having the classes of documents
    # with the same content
    duplicateFiles = []
    #comparison of the documents
    for file_x in files:
        if_dupl = False
        for class_ in duplicateFiles:
            # Comparing files having same content using cmp()
            # class_[0] represents a class having same content
            if_dupl = filecmp.cmp(
                file_x,
                class_[0],
                shallow=False
            )
            if if_dupl:
                class_.append(file_x)
                break
        if not if_dupl:
            duplicateFiles.append([file_x])
    #Print results
    if print_:
        print('Unique Files: {0} of {1}'.format(len(duplicateFiles), len(files)))
    #return
    return duplicateFiles

#def check if the same
def check_dup(df, col = 'InputFilePath'):
    df_inp = df[~df[col].isnull()]
    df_nan = df[df[col].isnull()]
    #list
    ls_files = df_nan[col].to_list()
    #processing
    if ls_files:
        with multiprocessing.Pool() as pool:
            _ = pool.map(remove_file, ls_files)
    #list
    files = df_inp[col].to_list()
    #file pairs
    file_pairs = list(itertools.combinations(files, 2))
    #init
    pooler = multiprocessing.Pool()
    #apply
    outputs = pooler.map(check_files, file_pairs)
    #set None
    pooler = None
    #compress identify which samples are repeating
    idxs2 = list(compress(range(len(outputs)), outputs))
    #list
    out = [file_pairs[idx] for idx in idxs2]
    #chain
    ls_out = list(set(list(itertools.chain(*out))))
    #dataframe1 which is unique
    df1 = df_inp[~df_inp[col].isin(ls_out)]
    #label df1
    df1['UniqueImage'] = np.arange(0, len(df1))
    #dataframe2
    df2 = df_inp[df_inp[col].isin(ls_out)]
    #get duplicate files
    ls_dup = get_dup(ls_out)
    #init
    ls_im = []
    #check
    for img_path in df2[col].to_list():
        #check
        for i, dup_file in enumerate(ls_dup):
            #set
            if img_path in dup_file:
                ls_im.append(i)
                break
    #set to unique image
    df2['UniqueImage'] = ls_im
    #add
    df2['UniqueImage'] = df2['UniqueImage'] + len(df1)
    #concat
    df = pd.concat([df1, df2]).reset_index(drop = True)
    #Print results
    print('Unique Files: {0} of {1}'.format(int(df['UniqueImage'].nunique()),
                                            len(df)))
    #return
    return df

#def check if the same
def check2dup(df, col = 'InputFilePath'):
    #split
    df_inp = df[~df[col].isnull()]
    df_nan = df[df[col].isnull()]
    #list
    #ls_files = df_nan[col].to_list()
    #processing
    #if ls_files:
    #    with multiprocessing.Pool() as pool:
    #        _ = pool.map(remove_file, ls_files)
    print('Missing Files {0} of {1}'.format(len(df_nan), len(df)))
    #dataframe 2
    ls_df2 = []
    #dataframe 3
    ls_df3 = []
    #iterate
    for pat in df_inp['PAT'].unique():
        #unique
        df_pat = df_inp[df_inp['PAT'] == pat]
        #check
        if len(df_pat) > 1:
            ls_df2.append(df_pat)
        else:
            ls_df3.append(df_pat)
    #concat multiple
    df_m = pd.concat(ls_df2).reset_index(drop = True)
    #concat single
    df_s = pd.concat(ls_df3).reset_index(drop = True)
    #unique image
    df_s['UniqueImage'] = np.arange(0, len(df_s))
    #inits
    pat_dfs = []
    #iterate
    for i, pat in enumerate(df_m['PAT'].unique()):
        #unique
        df_pat = df_m[df_m['PAT'] == pat]
        #list
        files = df_pat[col].to_list()
        #file pairs
        file_pairs = list(itertools.combinations(files, 2))
        #init
        pooler = multiprocessing.Pool()
        #apply
        outputs = pooler.map(check_files, file_pairs)
        #close
        pooler.close()
        #join
        pooler.join()
        #set None
        pooler = None
        #print
        #print(outputs)
        #compress identify which samples are repeating
        idxs2 = list(compress(range(len(outputs)), outputs))
        #list
        out = [file_pairs[idx] for idx in idxs2]
        #chain
        ls_out = list(set(list(itertools.chain(*out))))
        #dataframe1 which is unique
        df1 = df_pat[~df_pat[col].isin(ls_out)]
        #label df1
        df1['UniqueImage'] = np.arange(0, len(df1))
        #dataframe2
        df2 = df_pat[df_pat[col].isin(ls_out)]
        #get duplicate files
        ls_dup = get_dup(ls_out) 
        #init
        ls_im = []
        #check
        for img_path in df2[col].to_list():
            #check
            for j, dup_file in enumerate(ls_dup):
                #set
                if img_path in dup_file:
                    ls_im.append(j)
                    break
        #set to unique image
        df2['UniqueImage'] = ls_im
        #add
        df2['UniqueImage'] = df2['UniqueImage'] + len(df1)
        #concat
        df_c = pd.concat([df1, df2]).reset_index(drop = True)
        #unique images only
        df_c = df_c.drop_duplicates(['UniqueImage']) #unique image column is not useful
        #append
        pat_dfs.append(df_c)
        #break
        #if i == 2:
        #    break
    #combine
    df_d = pd.concat(pat_dfs).reset_index(drop = True)
    #out
    df_out = pd.concat([df_s, df_d]).reset_index(drop = True)
    #print
    print('Identified Repeating Files {0} of {1}'.format(len(df_out), len(df)))
    #check overlap #old dataframe - new dataframe
    ls_files = list(set(df['InputFilePath'].to_list()) - set(df_out['InputFilePath'].to_list()))
    #identify difference files
    if ls_files:
        #print
        print('Removed {0} Files'.format(len(ls_files)))
        #pool
        with multiprocessing.Pool() as pool:
            _ = pool.map(remove_file, ls_files)
    #return
    return df_out

#check duplicated data
def check2dup_data(df, save_folder, mode = 'CLR', col = 'InputFilePath',
                   subset_cols = ['Group', 'Laterality', 'Date', 'DateTime']):
    #clr
    df_mod = df[df['Mode'] == mode].reset_index(drop = True)
    #identifier
    df_mod['Identifier'] = df_mod.apply(app_, axis = 1, args = (subset_cols, ))
    #inputs
    df_inp = df_mod[~df_mod[col].isnull()]
    #nan
    df_nan = df_mod[df_mod[col].isnull()]
    #list
    ls_files = df_nan[col].to_list()
    #print
    print('Missing Files {0} of {1}'.format(len(df_nan), len(df_mod)))
    #processing
    #if ls_files:
    #    with multiprocessing.Pool() as pool:
    #        _ = pool.map(remove_file, ls_files)
    #duplicated mask
    duplicated_mask = df_inp.duplicated(subset = subset_cols, keep = False)
    #repeated rows
    df_inp_repeat = df_inp[duplicated_mask]
    #repeated rows
    df_inp_nonrepeat = df_inp[~duplicated_mask]
    #set unique image
    df_inp_nonrepeat['UniqueImage'] = np.nan
    #print
    print('Repeating Column Files {0} of {1}'.format(len(df_inp_repeat), len(df_inp)))
    #file_pairs
    file_pairs = []
    #iterate through the most relevant
    for id_ in df_inp_repeat['Identifier'].unique():
        #set based on column
        df_col = df_inp_repeat[df_inp_repeat['Identifier'] == id_]
        #list
        files = df_col[col].to_list()
        #file pairs
        file_pair = list(itertools.combinations(files, 2))
        #extend
        file_pairs.extend(file_pair)
    #init
    pooler = multiprocessing.Pool()
    #apply
    outputs = pooler.map(check_files, file_pairs)
    #set None
    pooler = None
    #compress identify which samples are repeating
    idxs2 = list(compress(range(len(outputs)), outputs))
    #list
    out = [file_pairs[idx] for idx in idxs2]
    #chain
    ls_out = list(set(list(itertools.chain(*out))))
    #dataframe1 which is unique
    df1 = df_inp_repeat[~df_inp_repeat[col].isin(ls_out)]
    #label df1
    df1['UniqueImage'] = np.arange(0, len(df1))
    #dataframe2
    df2 = df_inp_repeat[df_inp_repeat[col].isin(ls_out)]
    #get duplicate files
    ls_dup = get_dup(ls_out)
    #init
    ls_im = []
    #check
    for img_path in df2[col].to_list():
        #check
        for i, dup_file in enumerate(ls_dup):
            #set
            if img_path in dup_file:
                ls_im.append(i)
                break
    #set to unique image
    df2['UniqueImage'] = ls_im
    #add
    df2['UniqueImage'] = df2['UniqueImage'] + len(df1)
    #concat
    df3 = pd.concat([df1, df2]).reset_index(drop = True)
    #Print results
    print('Checked Unique Files: {0} of {1}'.format(int(df3['UniqueImage'].nunique()),
                                            len(df3)))
    #drop according to unique image
    df3 = df3.drop_duplicates(subset = ['UniqueImage'])
    #combine with nonrepeating rows
    df4 = pd.concat([df3, df_inp_nonrepeat]).reset_index(drop = True)
    #check overlap #old dataframe - new dataframe
    ls_files = list(set(df_inp['InputFilePath'].to_list()) - set(df4['InputFilePath'].to_list()))
    #identify difference files
    if ls_files:
        #print
        print('Removed {0} Files'.format(len(ls_files)))
        #pool
        with multiprocessing.Pool() as pool:
            _ = pool.map(remove_file, ls_files)
    #save mode
    df4.to_pickle(save_folder + 'df_' + mode.lower() + '_dataset.pkl')
    #print results
    print('Dataset Files: {0} of {1}'.format(len(df4), len(df_mod)))
    #return
    return df4

#remove files with 
def remove_files_with_extension(folder_path, extension):
    """Removes all files with the specified extension in the given folder.

    Args:
        folder_path: The path to the folder.
        extension: The file extension to remove (e.g., ".txt", ".pdf").
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            #print(f"Removed: {file_path}")

#app_
def app_(row, subset_cols):
    return ''.join(str(row[col]) for col in subset_cols)

#the characteristics of the data
def small_data(df, label):
    #select a subset of the data based on a label
    df = df[~df[label].isnull()]
    #postive samples
    df_pos = df[df[label] == 1]
    #negative samples
    df_neg = df[df[label] == 0]
    #return
    return df_new

#bucketname
bucket_name = ''
#save_folder
save_folder = '/'
#rm_dataset (remove previous files (.nii.gz) in dataset folder)
rm_dataset = True
#subset cols
subset_cols = ['Group', 'Laterality', 'Date', 'DateTime']
#label
label = 'Class'

#main
if __name__ == '__main__':
    #dcm folder
    dcm_folder = 'dcm/'
    #dcm folder
    if os.path.isdir(dcm_folder) == False:
        os.mkdir(dcm_folder)
    #dcm folder
    dcm_folder = 'dcm/'
    #dcm folder
    if os.path.isdir(dcm_folder) == False:
        os.mkdir(dcm_folder)
    #remove files
    if rm_dataset:
        remove_files_with_extension(save_folder, '.nii.gz')
    #read pickle
    df_img = pd.read_pickle(save_folder + 'df_data.pkl')
    #subset
    #df_img = pd.concat([df_img[df_img['Mode'] == 'CLR'].head(n = 10).reset_index(drop = True), 
    #                    df_img[df_img['Mode'] == 'AF'].head(n = 10)]).reset_index(drop = True)
    #small data
    df_img = small_data(df_img, label)
    #records
    ls_df = df_img.to_dict('records')
    #init
    pooler = multiprocessing.Pool()
    #apply
    outputs = pooler.map(process_image, ls_df)
    #close
    pooler.close()
    #join
    pooler.join()
    #none
    pooler = None
    #get outputs
    df_outputs = pd.DataFrame(list(outputs))
    #concat
    df = pd.concat([df_img.reset_index(drop = True), df_outputs], axis = 1)
    #print
    print('Colour Processing')
    #check2dup_data
    df_clr = check2dup_data(df, save_folder, mode = 'CLR', col = 'InputFilePath',
                   subset_cols = subset_cols)