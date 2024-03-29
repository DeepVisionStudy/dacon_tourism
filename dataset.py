import re
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')


class CategoryDataset(Dataset):
    def __init__(self, text, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len, transform):
        self.text = text
        self.image_path = image_path
        self.cats1 = cats1
        self.cats2 = cats2
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
        self.transform = transform
	
    def __len__(self):
        return len(self.text)
	
    def __getitem__(self, item):
        text = str(self.text[item])
        # text = self._clean_text(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        image_path = osp.join(PATH_DATA, str(self.image_path[item])[2:])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        
        cat = self.cats1[item]
        cat2 = self.cats2[item]
        cat3 = self.cats3[item]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_feature['pixel_values'][0],
            'cats1': torch.tensor(cat, dtype=torch.long),
            'cats2': torch.tensor(cat2, dtype=torch.long),
            'cats3': torch.tensor(cat3, dtype=torch.long)
        }
    
    def _clean_text(self, text):
        text = re.sub('[^. \u3131-\u3163\uac00-\ud7a3]+', '', text) # only korean
        return text


def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, num_workers, mode):
    if mode == 'train':
        shuffle_ = True
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
    elif mode == 'valid':
        shuffle_ = False
        transform = None
        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        # ])
    ds = CategoryDataset(
		text=df.overview.to_numpy(),
		image_path=df.img_path.to_numpy(),
		cats1=df.cat1.to_numpy(),
		cats2=df.cat2.to_numpy(),
		cats3=df.cat3.to_numpy(),
		tokenizer=tokenizer,
		feature_extractor=feature_extractor,
		max_len=max_len,
        transform=transform,
	)
    return DataLoader(
		ds,
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=shuffle_,
        pin_memory=True
	)


class CategoryDataset_test(Dataset):
    def __init__(self, text, image_path, tokenizer, feature_extractor, max_len, transform):
        self.text = text
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
        self.transform = transform
	
    def __len__(self):
        return len(self.text)
	
    def __getitem__(self, item):
        text = str(self.text[item])
        # text = self._clean_text(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        image_path = osp.join(PATH_DATA, str(self.image_path[item])[2:])
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_feature['pixel_values'][0],
        }
    
    def _clean_text(self, text):
        text = re.sub('[^. \u3131-\u3163\uac00-\ud7a3]+', '', text) # only korean
        return text


def create_data_loader_test(df, tokenizer, feature_extractor, max_len, hflip=False, shuffle_=False):
    if hflip:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
        ])
    else:
        transform = None
    ds = CategoryDataset_test(
		text=df.overview.to_numpy(),
		image_path=df.img_path.to_numpy(),
		tokenizer=tokenizer,
		feature_extractor=feature_extractor,
		max_len=max_len,
        transform=transform,
	)
    return DataLoader(
		ds,
		batch_size=1,
		num_workers=4,
		shuffle=shuffle_,
	)
