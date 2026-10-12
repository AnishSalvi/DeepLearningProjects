#imports
import numpy as np
import os
import pandas as pd
import multiprocessing
import torch
import time
import argparse
#set the cpu count
cpu_count = multiprocessing.cpu_count() - 1
print('CPU Count: ', cpu_count)

#functions

#get files
def get_files(fpath):
    files = []
    for file in sorted(os.listdir(fpath)):
        if '.ipynb_checkpoints' in file or 'imagelist.txt' in file:
            continue
        else:
            files.append(file)
    #return
    return files

#get facts
def get_info(row, fpath):
    id_var = int(row['Filename'].rsplit('.')[0])
    image_path = fpath + row['Filename']
    return id_var, image_path

#the image generator class
class ImageGeneratorMT(torch.utils.data.Dataset):
    #init
    def __init__(self, df, batch_size):
        #shuffle
        self.df = df.sample(batch_size).reset_index(drop=True)
    #len
    def __len__(self):
        return len(self.df)
    #get item
    def __getitem__(self, idx):
        #get row
        row = self.df.iloc[idx]
        #read in the numpy file
        image = np.load(row['Image'])
        #get the id
        image_id = row['ID']
        #return
        return image, image_id

#main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Assignment')
    parser.add_argument('--fpath', type = str, default = 'images/', required = False)
    parser.add_argument('--num_iter', type = int, default = 3, required = False)
    parser.add_argument('--batch_size', type = int, default = 25, required = False) #number of samples to draw from
    parser.add_argument('--num_threads', type = int, default = 0, required = False)
    parser.add_argument('--batch_load', type = int, default = 15, required = False)
    args = parser.parse_args()
    
    #set
    fpath = args.fpath
    num_iter = args.num_iter
    batch_size = args.batch_size
    num_threads = args.num_threads
    batch_load = args.batch_load
    
    #start
    start_time = time.time()
    #main
    df = pd.DataFrame(get_files(fpath), columns = ['Filename'])
    #get facts about the files
    df['ID'], df['Image'] = zip(*df.apply(get_info, axis = 1, args = (fpath, )))
    #num samples
    num_samples = len(df)
    
    #iteration_ids
    iteration_ids = []
    #for one iteration
    for n_iter in range(num_iter):
        #dataset
        dset = ImageGeneratorMT(df, batch_size)
        #create the loader
        loader = torch.utils.data.DataLoader(dset, batch_size = batch_load, num_workers = num_threads, shuffle = False, drop_last = False)
        #load the data but only take the first batch
        image_ids = []
        #iterate
        for i, (image, image_id) in enumerate(loader):
            image_ids.append(image_id)
        #at the end of the run
        run_ids = torch.cat(image_ids).numpy()
        #append
        iteration_ids.append(run_ids)
        
    #post-process
    all_iterations = np.concatenate(iteration_ids)
    #get counts
    numbers, counts = np.unique(all_iterations, return_counts = True)
    #print
    print('Used {0} of {1} Images or {2:.2g}'.format(len(numbers), num_samples, len(numbers) / num_samples))       
    print("--- %s seconds ---" % (time.time() - start_time))


    
    

