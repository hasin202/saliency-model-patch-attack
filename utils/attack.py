from utils.loss_function import SaliencyLoss
import torch
import cv2


def attack(model, img, target_saliency, target_mask, loss_target, lr):
    loss_fn = SaliencyLoss()
    loss, iterations = 0, 0
    original_img = torch.clone(img)
    while loss_target > loss:

        print('------------------------------------------')

        img = img.clone().detach().requires_grad_(True)
        img.retain_grad()

        out = model(img)

        # loss = loss_fn(out,target_saliency,loss_type='kldiv')
        loss = loss_fn(out, target_saliency, loss_type='cc')
        loss.backward()
        print(torch.max(img.grad)*lr)
        img = img + (img.grad * lr)

        print(loss)
        print(lr)
        print(img.shape)

        if lr > 1:
            lr = lr * 0.65
            if lr < 1:
                lr = 1

        iterations = iterations + 1

    return img, iterations
