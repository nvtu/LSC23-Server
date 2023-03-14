import __init__
from PIL import Image
from clip_model import CLIP
from typing import List
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import argparse
import os
import glob


BATCH_INDEX_FOR_SAVING = 1
DEFAULT_DATASET_NAME = 'lsc23'
DEFAULT_DEVICE = 'cuda'
DEFAULT_MODEL_NAME = 'ViT-L/14'
DEFAULT_BATCH_SIZE = 512
DEFAULT_START_INDEX = 0

class ImageDataset(Dataset):

    def __init__(self, dataset_indices: List[str], transform, device: str = DEFAULT_DEVICE):
        self.dataset_indices = dataset_indices
        self.transform = transform
        self.device = device
    
    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, index: int):
        item = self.dataset_indices[index]
        image = self.transform(Image.open(item)).to(self.device)
        return image, item


def get_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type = str)
    parser.add_argument('output_folder_path', type = str)
    parser.add_argument('--dataset_name', type = str, default = DEFAULT_DATASET_NAME)
    parser.add_argument('--device', type = str, default = DEFAULT_DEVICE)
    parser.add_argument('--model_name', type = str, default = DEFAULT_MODEL_NAME)
    parser.add_argument('--batch_size', type = int, default = DEFAULT_BATCH_SIZE)
    parser.add_argument('--start_index', type = int, default = DEFAULT_START_INDEX)

    args = parser.parse_args()
    return args


def create_folder(path: str):
    """
    Utilities to create new folder
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataset_file_indices(dataset_path: str, extension: str = ".jpg", cache: bool = True) -> List[str]:
    """
    Get all the items from the dataset and return the list of item paths
    """
    dataset_indices = []
    index_file_path = os.path.abspath('./dataset_indices.txt')

    # Check if we have a cached version of the indices
    # If we do, load it and return it
    if os.path.exists(index_file_path):
        dataset_indices = [line.strip() for line in open(index_file_path, 'r').readlines()]
    else:
        # If we don't, create a new one 
        # Search for all the files in the dataset with a given extension recursively
        pattern = f'{dataset_path}/**/*{extension}'
        for file in glob.glob(pattern, recursive = True):
            dataset_indices.append(file)

        # Save the indices to a file if we want to cache it
        if cache == True:
            with open(index_file_path, 'w') as f:
                for item in dataset_indices:
                    print(item, file = f)

    return dataset_indices


def save(dataset_name: str, features, feature_indices, iter_index: int):
    # Save the tensor features and their indices
    feature_output_path = os.path.join(args.output_folder_path, f'{dataset_name}_features_{iter_index}.pt')
    torch.save(features, feature_output_path)

    feature_indices_path = os.path.join(args.output_folder_path, f'{dataset_name}_indices_{iter_index}.pt')
    torch.save(feature_indices, feature_indices_path)


def post_processing(features_folder_path: str, dataset_name: str):
    # Concatenate all the saved features and indices
    features = []
    feature_indices = []

    for file in glob.glob(f'{features_folder_path}/{dataset_name}_features_*.pt'):
        corresponding_indices_file = file.replace('features', 'indices')
        feats = torch.load(file)
        features.append(feats)
        feature_indices += torch.load(corresponding_indices_file)
    
    features = torch.cat(features, dim=0)

    # Save the final features and indices
    model_name = DEFAULT_MODEL_NAME.replace('/', '-')
    feature_output_path = os.path.join(features_folder_path, f'{dataset_name}_{model_name}_embeddings.pt')
    torch.save(features, feature_output_path)

    feature_indices_path = os.path.join(features_folder_path, f'{dataset_name}_{model_name}_indices.pt')
    torch.save(feature_indices, feature_indices_path)


def run(args: argparse.Namespace):
    # Initialize CLIP model
    clip_model = CLIP(args.model_name, args.device)

    # Get path indices of all of the items in the dataset
    dataset_indices = get_dataset_file_indices(args.dataset_path)

    # Get current starting index in case previous ones were processed
    index = args.start_index
    if index > 0: 
        real_index = index * args.batch_size
        dataset_indices = dataset_indices[real_index:]

    dataset = ImageDataset(dataset_indices, clip_model.preprocess, device = args.device)
    # Encode entire dataset
    features = []
    feature_indices = []
    for images, paths in tqdm(DataLoader(dataset, batch_size = args.batch_size)):
        index += 1

        image_features = clip_model.encode_image(images)
        feat = image_features.detach().cpu()
        features.append(feat)
        feature_indices += paths

        _, r = divmod(index, BATCH_INDEX_FOR_SAVING)
        if r == 0:
            features = torch.cat(features, dim=0)
            save(args.dataset_name, features, feature_indices, index)
            # Reset features and indices
            features = []
            feature_indices = []

    # Save the last batch in case it's not a multiple of BATCH_INDEX_FOR_SAVING
    q, r = divmod(index, BATCH_INDEX_FOR_SAVING)
    if r > 0:
        index = (q + 1) * BATCH_INDEX_FOR_SAVING
        features = torch.cat(features, dim=0)
        save(args.dataset_name, features, feature_indices, index)


if __name__ == '__main__':
    args = get_arguments()
    create_folder(args.output_folder_path)
    
    run(args)

    print('Post-processing...')
    post_processing(args.output_folder_path, args.dataset_name)