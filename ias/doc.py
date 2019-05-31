import os
import numpy as np
import scipy.io as sio
import torch
import ipdb
from PIL import Image
from torch.utils import data

num_classes = 23
ignore_label = 255


'''
0=background, 1=text, 2=header, 3=figure, 4=list, 5=tablecaption # 6=tablebody, 7=formula, 7=matrix, 8=tablefootnote, 1=footer, 
i'''
palette = [255,255,255,0,0,255,0,255,0,0,255,255,255,0,0,255,0,255,255,255,0,125,125,125,255,0,125,125,0,255,125,225,0,225,125,0,0,125,225,0,225,125,200,100,50,200,50,100,100,200,50,100,50,200,50,200,100,50,100,200,87,193,224,224,193,87,193,87,224]



zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
	palette.append(0)
def colorize_mask(mask):
	# mask: numpy array of the mask
	new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
	new_mask.putpalette(palette)
	return new_mask
def colorize_mask_combine(mask,img_path):
	new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
	org_image = Image.open(os.path.join(img_path)).convert('RGB')
	new_mask.putpalette(palette)
	mask_combine = Image.blend(new_mask.convert("RGB"),org_image,0.5)
	return mask_combine
def colorize_mask_combine_eval(mask,img_path):
	new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
	org_image = Image.open(os.path.join(img_path)).convert('RGB')
	org_image = org_image.resize(new_mask.size, Image.ANTIALIAS)
	new_mask.putpalette(palette)
	mask_combine = Image.blend(new_mask.convert("RGB"),org_image,0.5)
	return mask_combine,org_image


def make_dataset(mode,root):
	assert mode in ['train', 'val', 'test_eva', 'test','eval']
	items = []
	if mode == 'eval':
		data_list = os.listdir(root)
		for it in data_list:
			items.append(os.path.join(root, it))
	elif mode == 'train':
		#load page images from marmot table detection dataset english
		img_path = os.path.join(root, 'data', 'img')
		mask_path = os.path.join(root, 'data', 'ind')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'data','train.txt')).readlines()] 
		
		for it in data_list:
			item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0] + '.png'))
			items.append(item)
				
	elif mode == 'val':

		#load page images from marmot table detection dataset english
		img_path = os.path.join(root, 'data', 'img')
		mask_path = os.path.join(root, 'data', 'ind')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'data','val.txt')).readlines()] 
		
		for it in data_list:
			item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0]+ '.png'))
			items.append(item)

	elif mode == 'test_eva':

		img_path = os.path.join(root, 'data', 'img')
		mask_path = os.path.join(root, 'data', 'ind')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'data','test.txt')).readlines()] 
		
		for it in data_list:
			item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0]+ '.png'))
			items.append(item)
	else:
		#load page images from marmot table detection dataset english
		img_path = os.path.join(root, 'data', 'img')
		mask_path = os.path.join(root, 'data', 'ind')
		data_list = [l.strip('\n') for l in open(os.path.join(
			root, 'data','test.txt')).readlines()] 
		
		for it in data_list:
			item = (os.path.join(img_path, it))
			items.append(item)
				   
	return items   

 

class DOC(data.Dataset):
	def __init__(self, mode,root, joint_transform=None, transform=None, target_transform=None,scaleMinSide=None):
		self.imgs = make_dataset(mode,root)
		if len(self.imgs) == 0:
			raise RuntimeError
		self.mode = mode
		self.scaleMinSide = scaleMinSide
		self.joint_transform = joint_transform
		self.transform = transform
		self.target_transform = target_transform

	   
	def __getitem__(self, index):
		if self.mode == 'test':
			img_path  = self.imgs[index]
			img_name = img_path.split('/')[-1]
			#a,b,c,d,e,f,img_name = img_path.split('/')
			img = Image.open(os.path.join(img_path)).convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
			return img_name, img
		if self.mode == 'eval':
			img_path  = self.imgs[index]
			img_name = img_path.split('/')[-1]
			img = Image.open(os.path.join(img_path)).convert('RGB')
			factor =  float(self.scaleMinSide)/np.min(img.size)
			img = img.resize((int(img.width * factor), int(img.height * factor)))
			if self.transform is not None:
				img = self.transform(img)
			return img_name, img
								  
								  
				 
		img_path, mask_path = self.imgs[index]
		img = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path).convert('RGB')
		imgarr = np.array(mask)
		imgarr1 = imgarr[:, :, 0]
		mask = Image.fromarray(imgarr1).convert('P')

				
		if self.joint_transform is not None:
			img, mask = self.joint_transform(img, mask)
			

		if isinstance(img, list) and isinstance(mask, list):
			if self.transform is not None:
				img = [self.transform(e) for e in img]
			if self.target_transform is not None:
				mask = [self.target_transform(e) for e in mask]
			img, mask = torch.stack(img, 0), torch.stack(mask, 0)
			
		else:
			if self.transform is not None:
				img = self.transform(img)
				
			if self.target_transform is not None:
				mask = self.target_transform(mask)
				
		return img, mask

	def __len__(self):
		return len(self.imgs)
