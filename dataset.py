import cv2
import random
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader

PATH_BASE = './'
PATH_DATA = osp.join(PATH_BASE, 'data')


class CategoryDataset(Dataset):
    def __init__(self, text, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.cats1 = cats1
        self.cats2 = cats2
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
	
    def __len__(self):
        return len(self.text)
	
    def __getitem__(self, item):
        text = str(self.text[item])
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if random.random() >= 0.5:
            image = cv2.flip(image, 1) # horizontal flip
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


def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, num_workers, shuffle_=False):
	ds = CategoryDataset(
		text=df.overview.to_numpy(),
		image_path=df.img_path.to_numpy(),
		cats1=df.cat1.to_numpy(),
		cats2=df.cat2.to_numpy(),
		cats3=df.cat3.to_numpy(),
		tokenizer=tokenizer,
		feature_extractor=feature_extractor,
		max_len=max_len
	)
	return DataLoader(
		ds,
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=shuffle_,
        pin_memory=True
	)


class CategoryDataset_test(Dataset):
    def __init__(self, text, image_path, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
	
    def __len__(self):
        return len(self.text)
	
    def __getitem__(self, item):
        text = str(self.text[item])
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_feature['pixel_values'][0],
        }


def create_data_loader_test(df, tokenizer, feature_extractor, max_len, shuffle_=False):
	ds = CategoryDataset_test(
		text=df.overview.to_numpy(),
		image_path=df.img_path.to_numpy(),
		tokenizer=tokenizer,
		feature_extractor=feature_extractor,
		max_len=max_len
	)
	return DataLoader(
		ds,
		batch_size=1,
		num_workers=4,
		shuffle=shuffle_,
	)
