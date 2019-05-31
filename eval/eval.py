import datetime
import cv2
import pytesseract
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image, ImageDraw,ImageOps
import os
import ipdb
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import sys
import numpy as np
from skimage import measure
from subprocess import call
sys.path.insert(0, '../')
from utils import joint_transforms as simul_transforms
from utils import transforms as extended_transforms
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from ias import doc
import argparse
parser = argparse.ArgumentParser(description='Train document image segmentation')
parser.add_argument('-u','--user', type=str, default='jobinkv',
    help='user id in ada')
parser.add_argument('-e','--exp', type=str, default='exp1',
    help='name of output folder')
parser.add_argument('-d','--dataset', type=str, default='ias',
    help='choose the dataset: cvpr(9 labels) or dsse(7 labels)')
parser.add_argument('-n','--net', type=str, default='psp',
    help='choose the network architecture: psp or mfcn')
#parser.add_argument('-s','--snapshot', type=str, default='',
parser.add_argument('-s','--snapshot', type=str, default='pspResnet152V0.0.pth',
    help='give the trained model for further training')
parser.add_argument('-l','--log', type=str, default='4000',
    help='give the folder name for saving the tensorflow logs')
parser.add_argument('-m','--model', type=str, default='resnet152',
    help='resnet101,resnet152,resnet18,resnet34,resnet50')
args = parser.parse_args()
print('The exp arguments are ',args.user,args.exp,args.net,args.dataset)

#ckpt_path = '/home/crisp/w/'#input folder
ckpt_path = '../outFiles/'
exp_name = args.log #output folder
if not os.path.exists(os.path.join(ckpt_path, exp_name)):
    os.makedirs(os.path.join(ckpt_path, exp_name))
dataset = args.dataset
jobid=args.log
network = args.net
snapShort=args.snapshot
#Dataroot = '/home/crisp/w/dataset_v3/augmented' #location of data
Dataroot = '../inputImages/' #location of data
root1 = '../trainedModel/'#location of pretrained model
if args.model=='resnet101':
    resnet = models.resnet101()
    res_path = os.path.join(root1,    'resnet101-5d3b4d8f.pth')
if args.model=='resnet152':
    resnet = models.resnet152()
    res_path = os.path.join(root1,    'resnet152-b121ed2d.pth')
if args.model=='resnet18':
    resnet = models.resnet18()
    res_path = os.path.join(root1,    'resnet18-5c106cde.pth')
if args.model=='resnet34':
    resnet = models.resnet34()
    res_path = os.path.join(root1,    'resnet34-333f7ec4.pth')
if args.model=='resnet50':
    resnet = models.resnet50()
    res_path = os.path.join(root1,    'resnet50-19c8e357.pth')
args = {
    'train_batch_size': 4,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'max_iter':40e3,
        'train_img_side':1024,# images are mapped to minimum side.
    'input_size': 512,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot':snapShort,  # empty string denotes learning from scratch
    'print_freq': 100,
    'max_epoch':5,
    'dataset':dataset,
    'network':network,
    'jobid':jobid,
    'No_train_images':0,
    'Type_of_train_image':'',
    'Auxilary_loss_contribution':0.5,
    'Pretrained_Model':args.model
}
def colorMap(indx):
    colors=[(255,255,255),(0,0,255),(0,255,0),(0,255,255),(255,0,0),(255,0,255),(255,255,0),(125,125,125),
            (255,0,125),(125,0,255),(125,225,0),(225,125,0),(0,125,225),(0,225,125),(200,100,50),(200,50,100),
    (100,200,50),(100,50,200),(50,200,100),(50,100,200),(224,193,87),(87,193,224),(224,87,193)]
    if indx<23:
        return colors[indx]
    else:
        return (0,0,0)

def labelMap(indx):
    labels = ['background','invoice_num', 'invoice_date', 'po_num', 'po_date', 'sup_name', 'rec_name', 'sup_add', 'rec_add', 'sup_gst', 'rec_gst', 'sup_pan', 'tot_num', 'tot_word', 'tot_tax_num', 'tot_tax_word', 'cgst_amt', 'sgst_amt', 'cgst_per', 'sgst_per', 'rec_pan', 'igst_amt', 'igst_per']
    if indx<23:
        return labels[indx]
    else:
        return 'un known'


def main(train_args):
    '''
    This function read all images in a folder "Dataroot" 
    and detect the key regions such as total amount, addres, etc.
    and save this value to a text file with base name equal to the 
    image file name.
    '''
    print(doc.num_classes)
    # load the network architecture and trained model
    net = PSPNet(num_classes=doc.num_classes,resnet=resnet,res_path=res_path,pretrained=False).cuda()
    print("number of cuda devices = ", torch.cuda.device_count())
    print('training resumes from ' + train_args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(root1, train_args['snapshot'])))
    net.eval()
    mean_std =([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])
    test_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    # dataloader
    test_set = doc.DOC('eval',Dataroot,transform=test_input_transform,scaleMinSide=train_args['train_img_side'])
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)
    # read each image
    # visualisation
    visual = False
    for vi, data in enumerate(test_loader):
        img_name, img = data
        img_name = img_name[0]
        with torch.no_grad():
            img = img.cuda()
            output = net(img)
            # get predicted segmented regions
            prediction1 = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            prediction,orgImg = doc.colorize_mask_combine_eval(prediction1,Dataroot+img_name)
            if visual:
                orgImg_cv = np.array(orgImg) 
                orgImg_cv = orgImg_cv[:, :, ::-1].copy()
                orgImg_cv = cv2.copyMakeBorder(orgImg_cv,0,0,0,200,cv2.BORDER_CONSTANT,value=[0,0,0])
                orgImg_cvBac = orgImg_cv.copy()
            out_file = open(os.path.join(ckpt_path, exp_name, img_name+'.txt' ), "w")
            for label in np.unique(prediction1):
                if label==0:
                    continue
                blobs = prediction1==label
                # filter out thr irelevent regions
                boxx = regionFilter(blobs)
                # get the ocr of selected regions
                value = blobOcr(orgImg,blobs,boxx)
                # print and write to the file
                print (labelMap(label),' ::: ',value)
                out_file.write(labelMap(label)+'\t:'+value+'\n')
                if visual:
                    xmin = boxx[1]
                    ymin = boxx[0]
                    xmax = boxx[3]
                    ymax = boxx[2]
                    cv2.rectangle(orgImg_cv,(xmin,ymin),(xmax,ymax),colorMap(label),-1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 1
                    text_size = cv2.getTextSize(labelMap(label), font, font_scale, thickness)
                    text_width = text_size[0][0]
                    text_height = text_size[0][1]
                    pt3 = (xmax,ymin-5)
                    pt4 = (xmax+text_size[0][0]+5,
                        ymin+text_size[0][1]+5)
                    pt5 = (xmax,ymin+text_size[0][1]-2)
                    cv2.rectangle(orgImg_cv,pt3,pt4,(0,0,0),-1)
                    cv2.putText(orgImg_cv,labelMap(label),pt5, font, font_scale, (0,255,255), thickness, 2)
            out_file.close()
            if visual:
                alpha = 0.5
                beta = (1.0 - alpha)
                blended = cv2.addWeighted(orgImg_cv, alpha, orgImg_cvBac, beta, 0.0)
                cv2.imwrite(os.path.join(ckpt_path, exp_name, img_name ),blended[:, :, ::-1])
def blobOcr(Image,blobs,boxx):
    '''
    Get the OCR value for all detected regions
    with the help of PyTessBaseAPI and pytesseract
    '''
    with PyTessBaseAPI() as api:
        xmin = max(boxx[1]-100,0)
        ymin = max(boxx[0]-100,0)
        xmax = min(boxx[3]+100,blobs.shape[1]-1)
        ymax = min(boxx[2]+100,blobs.shape[0]-1)
        
        #draw = ImageDraw.Draw(Image)
        crpBlob = blobs[ymin:ymax,xmin:xmax]
        crpImage = Image.crop((xmin,ymin,xmax,ymax))
        api.SetImage(crpImage)
        outString=''
        blocks = api.GetComponentImages(RIL.TEXTLINE, True)
        for i, (im, box, _, _) in enumerate(blocks):
            lineP = crpBlob[box['y']:(box['y']+box['h']),box['x']:(box['x']+box['w'])]
            if np.sum(lineP)>0:
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                im = ImageOps.expand(im,border=50,fill='white')
                outString = outString+(pytesseract.image_to_string(im))
    return outString 
def regionFilter(blobs):
    '''
    select the biggest region
    merge if small region are ocure together
    '''
    all_regions = measure.label(blobs)
    properties = measure.regionprops(all_regions)
    if len(properties)==1:
        return properties[0].bbox
    else:
        xmin = 100000
        xmax = 0
        ymin = 100000
        ymax = 0
        for ele in properties:
            if xmin>ele.bbox[1]:
                xmin = ele.bbox[1]
            if ymin>ele.bbox[0]:
                ymin = ele.bbox[0]
            if xmax<ele.bbox[3]:
                xmax = ele.bbox[3]
            if ymax<ele.bbox[2]:
                ymax = ele.bbox[2]
        temps = blobs[ymin:ymax,xmin:xmax]
        if float(np.sum(temps))/temps.size>=0.3:
            return (ymin,xmin,ymax,xmax)
    area = [ele.area for ele in properties]
    largest_blob_ind = np.argmax(area)
    return properties[largest_blob_ind].bbox
if __name__ == '__main__':
    main(args)
