from src.tests import *
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms
import time


def foreground(img1_path, mask_path):
	foreground = Image.open(img1_path).convert("RGB")
	process_foreground = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
	model.eval()

	input_layer = process_foreground(foreground)
	input_batch = input_layer.unsqueeze(0) # create a mini-batch as expected by the model

	with torch.no_grad():
	    output = model(input_batch)['out'][0]
	output_layer = output.argmax(0)

	# create a color pallette, selecting a color for each class
	palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
	colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
	colors = (colors % 255).numpy().astype("uint8")

	# plot the semantic segmentation predictions of 21 classes in each color
	r = Image.fromarray(output_layer.byte().cpu().numpy()).resize(foreground.size)
	r.putpalette(colors)
	bw = r.convert('L')
	bw.save(mask_path)

	# crop out object
	src1 = cv2.imread(img1_path)
	src1_mask = cv2.imread(mask_path)
	_,thresh1  = cv2.threshold(src1_mask,30,255,cv2.THRESH_BINARY)
	src1 = cv2.cvtColor(src1,cv2.COLOR_RGB2BGR)
	src1_mask = cv2.cvtColor(thresh1,cv2.COLOR_RGB2BGR)

	mask_out=cv2.subtract(src1_mask,src1)
	mask_out=cv2.subtract(src1_mask,mask_out)
	#cv2.imwrite(img1_path,cv2.cvtColor(mask_out, cv2.COLOR_RGB2BGR))

	# create mask
	src = cv2.imread(mask_path, 1)
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	thr,alpha = cv2.threshold(tmp,30,255,cv2.THRESH_BINARY)
	b, g, r = cv2.split(src)
	rgba = [b,g,r, alpha]
	dst = cv2.merge(rgba,4)
	print(dst.shape)
	cv2.imwrite(mask_path, dst)

	plt.figure(figsize=(10, 10))
	plt.imshow(dst)

if __name__ == "__main__": 
	foreground()