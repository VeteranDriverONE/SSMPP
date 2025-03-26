import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def calculate_metric_parcase2(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    # image, label shape: slice,H,W
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_model(model, test_loader):
    
    model.eval()
    dice_ls = []
    jc_ls = []
    hd_ls = []
    asd_ls = []
    for sample in test_loader:
        images = sample['image'].float().cuda()
        labels = sample['label']
        with torch.no_grad():
            preds = model(images)

        preds = torch.softmax(preds, dim=1)
        preds_am = torch.argmax(preds, dim=1)
        dice, jc, hd, asd = calculate_metric_parcase2(preds_am.squeeze().cpu().numpy(), labels.squeeze().numpy())
        dice_ls.append(dice)
        jc_ls.append(jc)
        hd_ls.append(hd)
        asd_ls.append(asd)
    

    dice_np = np.array(dice_ls)
    jc_np = np.array(jc_ls)
    hd_np = np.array(hd_ls)
    asd_np = np.array(asd_ls)
    print('dice mean:{},std:{}\nJaccard mean:{},std:{}\nHD mean:{},std:{}\nASD mean:{},std:{},'.format(
            dice_np.mean(), dice_np.std(), jc_np.mean(), jc_np.std(), hd_np.mean(), hd_np.std(), asd_np.mean(), asd_np.std()))

    model.train()
    
    return dice_np, jc_np, hd_np, asd_np

def test_model_ds(model, test_loader):
    
    model.eval()
    dice_ls = []
    jc_ls = []
    hd_ls = []
    asd_ls = []
    for sample in test_loader:
        images = sample['image'].float().cuda()
        labels = sample['label']
        with torch.no_grad():
            preds, _, _, _ = model(images)

        preds = torch.softmax(preds, dim=1)
        preds_am = torch.argmax(preds, dim=1)
        dice, jc, hd, asd = calculate_metric_parcase2(preds_am.squeeze().cpu().numpy(), labels.squeeze().numpy())
        dice_ls.append(dice)
        jc_ls.append(jc)
        hd_ls.append(hd)
        asd_ls.append(asd)
    

    dice_np = np.array(dice_ls)
    jc_np = np.array(jc_ls)
    hd_np = np.array(hd_ls)
    asd_np = np.array(asd_ls)
    print('dice mean:{},std:{}\nJaccard mean:{},std:{}\nHD mean:{},std:{}\nASD mean:{},std:{},'.format(
            dice_np.mean(), dice_np.std(), jc_np.mean(), jc_np.std(), hd_np.mean(), hd_np.std(), asd_np.mean(), asd_np.std()))

    model.train()
    
    return dice_np, jc_np, hd_np, asd_np