# LifeInsight Server Demo for LSC'23

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## CLIP server
**CLIP server** is used to encode the query text/image into a targeted vector space. 

By default, the server employs the latest lightweight server -- **ViT-L/14** --  to encode text/image, which results in the feature vector with dimension of $1 \times 768$.

To change the model type, change the ```MODEL_NAME``` variable in the ```clip_server/.env``` file. Refer to the [CLIP repository](https://github.com/openai/CLIP/blob/main/clip/clip.py) to select the model of interest.

By default, the model is deployed in port 4001 by running the following commands:
```
$ cd clip_server
$ uvicorn main:app --reload --host 0.0.0.0 --port 4001
```

## CLIP Feature Extraction
To extract encoded CLIP features for an image dataset, run the following commands:
```
$ cd preprocess
$ python extract_clip_features $dataset_path $output_folder_path $ --dataset_name lsc23 --device cuda model_name ViT-L/14 --batch_size 512 --start_index 0
```

The final results are two files, which are encoded CLIP features and their indices with the name ```lsc23_ViT-L-14_embeddings.pt``` and ```lsc23_ViT-L-14_indices.pt```.
> :warning: Ensure that the CLIP model that you use is the same as the one used in the ```clip_server```
