import os 
def renameFiles(folder_type):
    train_or_val_or_test = os.path.join('dataset1',folder_type) 
    classes = os.listdir(train_or_val_or_test) 
    for classname in classes:
        dirname = os.path.join(train_or_val_or_test, classname) 
        for count, filename in enumerate(sorted(os.listdir(dirname))): 
            counter = count + 1
            dst = classname+'_%04d.jpg'%counter
            src =  filename
            os.rename(os.path.join(dirname, src), os.path.join(dirname,dst)) 
if __name__=='__main__':
    renameFiles('validation')
    renameFiles('test')
    


