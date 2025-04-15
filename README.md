# DSS EXPLAINABLE AI with D-RISE

This repository is an adaptation of the library [vision-explanation-methods](https://github.com/microsoft/vision-explanation-methods) of Microsoft to work with YOLO (Ultralytics).

In essence, D-RISE works by randomly masking the input images and isolating the parts that are most pertinent for the detection or segmentation of the object in question.

![alt text](image.png)

The code is setup to work for both CPU and GPU.

### This is an example of the result
![000000000139_chair_1](https://github.com/user-attachments/assets/a21dd8b5-1c11-4ea0-ac21-4a9fa48053a5)


#### Optional (just for GPUs)
If you have a GPU first you need to have CUDA installed
You can execute the following code to install it
```bash
bash install_nvidia.sh
```

## Environment creation
It's necessary to create the venv with the correct python version and packages
The following command will install python 3.10 and the main packages: ultralytics, vision-explanation-methods, opencv

```bash
bash env_setup.sh
```

## Example of execution

After the env setup is completed it is possible to run direcly the script to generate the images.

```bash
myenv/bin/python main.py --images_folder_path <IMG FOLDER PATH> --model_path <MODEL PATH> --output_folder_path <OUTPUT FOLDER PATH>
```


### Optional settings

There are others optional parameters:

- `conf_threshold` (float): Confidence threshold for displaying detections. (default: 0.5)

- `img_size` (int): Size to resize images for the model. (default: 640)

- `masks_num` (int): Number of masks to generate for D-RISE. More is slower but gives higher quality mask. (default: 500)

- `mask_res` (int): Resolution of the base mask. High resolutions will give finer masks, but more need to be run. (default: 8)

- `mask_padding` (int): Padding for the mask. (default: None)

To execute the script with any of the previously described parameters just run the following command adding the optional parameter.

```bash
myenv/bin/python main.py --images_folder_path <IMG FOLDER PATH> --model_path <MODEL PATH> --output_folder_path <OUTPUT FOLDER PATH> --masks_num <number like 1000> --mask_res <number like 8>
```

If you what to keep track of the resouces used, in another terminal you can run the following code

```bash
htop
```


## Test with kaggle dataset

First you need to have kaggle installed and the api-key placed in the folder `~/.kaggle/kaggle.json`

```bash
pip install kaggle
```

Then you can download the dataset using the following command

```bash
kaggle datasets download -d die9origephit/children-vs-adults-images
```

To extract the dataset and remove the zip you can execute this

```bash
python -c "
zip_file = 'children-vs-adults-images.zip'
folder = zip_file.split('.')[0]
import zipfile, os; zipfile.ZipFile(zip_file).extractall(folder); os.remove(zip_file)"
```

To extract 50 images from the folder you can use the following code and it will copy the images in another folder

```bash
python -c "
import os, random, shutil
source_folder = 'children-vs-adults-images'
destination_folder = 'sampled-images'
os.makedirs(destination_folder, exist_ok=True)
sample_size = 50  # Number of images to sample
images = []
for root, _, files in os.walk(source_folder):
    images.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
sampled_images = random.sample(images, min(sample_size, len(images)))
for img in sampled_images:
    shutil.copy(img, destination_folder)
print(f'Sampled {len(sampled_images)} images to {destination_folder}')
"
```

Then you can execute the D-RISE to the extracted images using this command

```bash
myenv/bin/python main.py --images_folder_path sampled-images --model_path yolov8n.pt --output_folder_path output
```



## Delete the environment

Once you are satisfied with the results and want to remove the environment to save space you can execute this code

```bash
rm -rf myenv
```
