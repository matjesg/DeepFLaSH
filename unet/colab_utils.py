from google.colab import files
from unet import utils
import os

############################################################
#  Upload files
############################################################
def upload_files():
    file_dict = files.upload()
    file_names = sorted(file_dict.keys())
    file_list = [utils.readImg(x, path='', channels=1, dimensions=1024) for x in file_names]
    return (file_names, file_list)

############################################################
#  Load sample images
############################################################
def load_samples(path, suffix):
  file_path = os.path.join('DeepFLaSH','sample_images',path)
  file_names = [fname for fname in sorted(os.listdir(file_path)) if fname.endswith(suffix + '.tif')]
  file_list = [utils.readImg(x, path=file_path, channels=1, dimensions = 1024) for x in file_names]
  return(file_names, file_list)




