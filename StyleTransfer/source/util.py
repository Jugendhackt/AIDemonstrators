import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import copy

from .nn import Normalization, ContentLoss, StyleLoss

from IPython import display

loader = transforms.Compose([
    transforms.Resize(512),  # scale imported image
    transforms.CenterCrop(512),
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def unload(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization_mean = torch.tensor(normalization_mean).to(device)
    normalization_std = torch.tensor(normalization_std).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn,
                       content_img, style_img, input_img, normalization_mean=[0.485, 0.456, 0.406], 
                       normalization_std=[0.229, 0.224, 0.225], num_steps=300,
                       style_weight=1000000, content_weight=1, content_layers=['conv_4'],
                       style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):                      
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    fig = plt.figure(figsize=(16, 10))
    fig.add_subplot(132).imshow(unload(content_img))
    plt.grid(False)
    plt.title("Content image")
    fig.add_subplot(133).imshow(unload(style_img))
    plt.grid(False)
    plt.title("Style image")
    ax = fig.add_subplot(131)

    fig.tight_layout()
    imsh = ax.imshow(unload(input_img))
    plt.grid(False)
    
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                display.clear_output(wait=True)
                # correcting image data to be between 0 and 1
                input_img.data.clamp_(0, 1)
                imsh.set_data(unload(input_img))
                display.display(fig)
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score, content_score))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img