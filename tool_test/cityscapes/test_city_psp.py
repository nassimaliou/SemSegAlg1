import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

import PIL.Image
import io

from model.pspnet import PSPNet, DeepLabV3
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize


cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--attack', action='store_true', help='evaluate the model with attack or not')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None

    global attack_flag
    if args.attack:
        attack_flag = True
    else:
        attack_flag = False

    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def M_PGD(input , gt , mask , model , eps , steps , alpha):
    input_unnorm = input.clone ().detach ()
    input_unnorm [:, 0, :, :] = input_unnorm [:, 0, :, :] * std_origin [0] + mean_origin [0]
    input_unnorm [:, 1, :, :] = input_unnorm [:, 1, :, :] * std_origin [1] + mean_origin [1]
    input_unnorm [:, 2, :, :] = input_unnorm [:, 2, :, :] * std_origin [2] +mean_origin [2]
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach ().clone ()
    gt = gt.detach ().clone ()
    adversarial_example = adversarial_example*mask
    gt = gt*mask

    adversarial_example.requires_grad = True

    for _ in range(steps):

        ignore_label = 255
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
        target /= 255

        # Forward pass
        output = model(adversarial_example)
        loss = criterion(output , target.detach ())

        # Backward pass
        model.zero_grad ()
        loss.backward ()

        # PGD step
        with torch.no_grad ():
            perturbation = alpha * adversarial_example.grad.sign()

            perturbed_image = adversarial_example + perturbation

            # Project perturbed image to epsilon -ball around original image
            perturbed_image = torch.max(torch.min(perturbed_image , clip_max), clip_min)
            perturbed_image = torch.clamp(perturbed_image , 0, 1)
            adversarial_example = perturbed_image

            adversarial_example [:, 0, :, :] = (adversarial_example [:, 0, :, :] - mean_origin [0]) / std_origin [0]
            adversarial_example [:, 1, :, :] = (adversarial_example [:, 1, :, :] - mean_origin [1]) / std_origin [1]
            adversarial_example [:, 2, :, :] = (adversarial_example [:, 2, :, :] - mean_origin [2]) / std_origin [2]
    return adversarial_example


def M_FGSM(input , gt , mask , model , clip_min , clip_max, eps):
    adversarial_example = input.detach ().clone ()
    gt = gt.detach ().clone ()
    adversarial_example.requires_grad = True
    adversarial_example = adversarial_example*mask
    gt = gt*mask

    model.zero_grad ()
    result = model(adversarial_example)

    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
    loss = criterion(result , gt.detach ())
    loss.backward ()
    res = input_variable.grad
    adversarial_example = input.detach ().clone ()

    adversarial_example [:, 0, :, :] = adversarial_example [:, 0, :, :] * std_origin [0] + mean_origin [0]
    adversarial_example [:, 1, :, :] = adversarial_example [:, 1, :, :] * std_origin [1] + mean_origin [1]
    adversarial_example [:, 2, :, :] = adversarial_example [:, 2, :, :] * std_origin [2] + mean_origin [2]

    adversarial_example = adversarial_example + eps * torch.sign(res)
    adversarial_example = torch.max(adversarial_example , clip_min)
    adversarial_example = torch.min(adversarial_example , clip_max)
    adversarial_example = torch.clamp(adversarial_example , min=0.0, max=1.0)

    adversarial_example [:, 0, :, :] = (adversarial_example [:, 0, :, :] - mean_origin [0]) / std_origin [0]
    adversarial_example [:, 1, :, :] = (adversarial_example [:, 1, :, :] - mean_origin [1]) / std_origin [1]
    adversarial_example [:, 2, :, :] = (adversarial_example [:, 2, :, :] - mean_origin [2]) / std_origin [2]

    return adversarial_example


def BIM(input, target, masked, model, eps=1, k_number=2, alpha=0.01):
    input_unnorm = input.clone().detach()
    input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * std_origin[0] + mean_origin[0]
    input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * std_origin[1] + mean_origin[1]
    input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * std_origin[2] + mean_origin[2]
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach().clone()
    adversarial_example.requires_grad = True
    for mm in range(k_number):
        adversarial_example = M_FGSM(adversarial_example, target, model, clip_min, clip_max, eps=alpha)
        adversarial_example = adversarial_example.detach()
        adversarial_example.requires_grad = True
        model.zero_grad()
    return adversarial_example


def main():
    global parameters
    global metrics
    global args, logger

    parameters = dict()
    metrics = dict()

    args = get_parser()
    logger = get_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.test_h - 1) % 8 == 0 and (args.test_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    global mean_origin
    global std_origin
    mean_origin = [0.485, 0.456, 0.406]
    std_origin = [0.229, 0.224, 0.225]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.test_list, transform=test_transform)
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    if not args.has_prediction:
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        start_time = time.time()
        test(test_loader, test_data.data_list, model, args.classes, mean, std, args.base_size, args.test_h, args.test_w, args.scales, gray_folder, color_folder, colors)
        end_time = time.time()
        duration = (end_time-start_time)/60
        metrics["Duration"] = duration


def net_process(model, image, target, masked, mean, std=None):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    masked = torch.from_numpy(masked.transpose((2, 0, 1))).float()
    target = torch.from_numpy(target).long()

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
        for t, m in zip(masked, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
        for t, m, s in zip(masked, mean, std):
            t.sub_(m).div_(s)

    input = input.unsqueeze(0).cuda()
    masked = masked.unsqueeze(0).cuda()
    target = target.unsqueeze(0).cuda()

    if attack_flag:
        flip = False
    else:
        flip = True

    if flip:
        input = torch.cat([input, input.flip(3)], 0)
        masked = torch.cat([masked, masked.flip(3)], 0)
        target = torch.cat([target, target.flip(2)], 0)
    if attack_flag:
        adver_input = M_PGD(input, target, masked, model, 0.03, 10, 0.01)
        with torch.no_grad():
            output = model(adver_input)
    else:
        with torch.no_grad():
            output = model(input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, target, masked, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        masked = cv2.copyMakeBorder(masked, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=255)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            masked_crop = masked[s_h:e_h, s_w:e_w].copy()
            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, target_crop, masked_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    start_eval = time.time()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        target = np.squeeze(target.numpy(), axis=0)

        #masking all classes except the selected one
        masked_input, target = apply_mask(input, target, 16)
        image = np.transpose(input, (1, 2, 0))
        masked_image = np.transpose(masked_input, (1, 2, 0))

        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        masked_image = cv2.resize(masked_image, (1024, 512), interpolation=cv2.INTER_LINEAR)
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            masked_image_scale = cv2.resize(masked_image, (1024, 512), interpolation=cv2.INTER_LINEAR)
            target_scale = cv2.resize(target.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            prediction += scale_process(model, image_scale, target_scale, masked_image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
        end_eval = time.time()

    metrics["Number of adv images"] = (i+1)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (1024, 512), interpolation=cv2.INTER_NEAREST)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

def intersection_over_union(output, target):
    ls = [7, 16, 26, 45, 46, 58, 70, 76, 84, 90, 108, 117, 119, 153, 164, 177, 194, 210] #cityscapes classes
    intersection = []
    union = []
    for cls in ls:
        output_mask = output == cls
        target_mask = target == cls

        intersection.append(np.logical_and(output_mask, target_mask).sum())
        union.append(np.logical_or(output_mask, target_mask).sum())
    iou = []
    for i in range(len(intersection)):
        if union[i] == 0:
            iou.append(0.0)
        else:    
            iou.append(intersection[i]/union[i])
    m_iou = iou
    while 0.0 in m_iou: m_iou.remove(0.0)
    if len(m_iou)!=0:
        mean_iou = sum(m_iou)/len(m_iou)
    else:
        mean_iou=0
    return (iou, mean_iou)

def apply_mask(input, target, mask=90):
    image = np.copy(input)
    mask_arr = (target == mask).astype(np.int)

    image[0] = image[0] * mask_arr
    image[1] = image[1] * mask_arr
    image[2] = image[2] * mask_arr

    target = target * mask_arr

    return image, target

def generate_mask(input , target , mask=120):

    image = np.copy(input)
    binary_mask = (target == mask).astype(np.int)

    return binary_mask

def resize_image(image_path, resize=None): 
    if isinstance(image_path, str):
        img = PIL.Image.open(image_path)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(image_path)))
        except Exception as e:
            data_bytes_io = io.BytesIO(image_path)
            img = PIL.Image.open(data_bytes_io)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()

if __name__ == '__main__':
    main()