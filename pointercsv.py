import pandas as pd
df=pd.read_csv('pointers/train2017.txt',names=['img'])
df['box']=df['img'].apply(lambda x: 'coco/labels/'+x.split('.')[0]+'.txt')
df.to_csv('pointer.csv')