import zipfile

" extracts directory into directory 
|_train
  |_images
  |_masks
|_test
  |_images
|_ depths.csv

  

url = "ttps://www.kaggle.com/c/10151/download-all"

with zipfile.ZipFile(url,"r") as zip_ref:
    zip_ref.extractall("targetdir")
