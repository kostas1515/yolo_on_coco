import os
import pandas as pd
import glob
import sys
from multiprocessing import Pool # for reading the CSVs faster



df = pd.DataFrame(columns=['class','xc','yc','w','h','filename'])
os.chdir('/mnt/data1/users/konsa15/workspace/notebooks/coco/labels/coco/labels/train2017')
print(os.getcwd())
files = glob.glob("*.txt")
dataset_len=len(files)
prg_counter=0
print(dataset_len)
for file in files:
    with open(file) as f:
        f=f.read()
        box=pd.DataFrame([map(float,x.split()) for x in f.rstrip('\n').split('\n')],columns=['class','xc','yc','w','h'])
        box['filename']=[file for x in f.rstrip('\n').split('\n')]
        df = df.append(box, ignore_index = True)
        
    sys.stdout.write('\rPgr:'+str(prg_counter/dataset_len*100)+'%')
    prg_counter+=1
    

# obj_idf=load_csvs() for parallel
obj_idf=(df['class'].value_counts(normalize=True).reset_index(name='obj_idf'))
new_df=df.groupby('filename')['class'].value_counts().reset_index(name='count')
img_idf=new_df['class'].value_counts(normalize=True).reset_index(name='img_idf')

obj_idf['img_idf']=img_idf['img_idf']

total_bins=df['class'].value_counts()
print(total_bins)
yc_idf=df.groupby('class')['yc'].sum()
xc_idf=df.groupby('class')['xc'].sum()

df['area']=df['w']*df['h']
area_idf=df.groupby('class')['area'].sum()


obj_idf['xc']=(xc_idf/total_bins)
obj_idf['yc']=(yc_idf/total_bins)
obj_idf['area']=(area_idf/total_bins)

# fig=obj_idf.plot(x='img_idf',y='obj_idf',kind='scatter').get_figure()
# fig.savefig('obj_img_idf-corr.png')
# obj_idf.corr()

#parallel implementation
def my_read_csv(filename):
    try:
        f=open(filename).read()
        box=pd.DataFrame([x.split() for x in f.rstrip('\n').split('\n')],columns=['class','xc','yc','w','h'])
        box['filename']=[filename for x in f.rstrip('\n').split('\n')]
        return box
    except FileNotFoundError:
        return None

def load_csvs():
    """Reads and joins all our CSV files into one big dataframe.
    We do it in parallel to make it faster, since otherwise it takes some time.
    Idea from: https://stackoverflow.com/questions/36587211/easiest-way-to-read-csv-files-with-multiprocessing-in-pandas
    
    """
    # set up your pool
    pool = Pool() 
    os.chdir('/mnt/data1/users/konsa15/workspace/notebooks/coco/labels/coco/labels/train2017')
    print(os.getcwd())
    files = os.listdir('.')
    file_list = [filename for filename in files if filename.split('.')[1]=='txt']
    print(len(file_list))
    df_list = pool.map(my_read_csv, file_list)
    # reduce the list of dataframes to a single dataframe
    return pd.concat(df_list, ignore_index=True)