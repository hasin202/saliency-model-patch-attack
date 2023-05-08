from utils.loss_function import SaliencyLoss
import torch
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attack(model, img, target_saliency, target_mask, loss_target, lr, mask):
    print(mask)
    loss_fn = SaliencyLoss()
    loss, iterations = 0, 0
    original_img = torch.clone(img)
    original_img = original_img.squeeze().permute(1, 2, 0).numpy()
    while loss_target > loss:

        print('------------------------------------------')

        img = img.clone().detach().requires_grad_(True)
        img.retain_grad()

        out = model(img)

        # loss = loss_fn(out,target_saliency,loss_type='kldiv')
        loss = loss_fn(out, target_saliency, loss_type='cc')
        loss.backward()

        img = img + (img.grad * lr)

        print(loss)
        print(lr)
        if mask == True:
            img = img.squeeze().detach().permute(1, 2, 0).numpy()
            mask = cv2.bitwise_and(
                img, img, mask=np.expand_dims(target_mask, axis=-1))
            original_img = cv2.bitwise_and(original_img, original_img, mask=cv2.bitwise_not(
                np.expand_dims(target_mask, axis=-1)))
            img = cv2.addWeighted(mask, 1, original_img, 1, 0)

            img = torch.from_numpy(img)
            img = img.type(torch.FloatTensor).to(device)
            img = torch.permute(img, (2, 0, 1)).unsqueeze(0)

        iterations = iterations + 1

    return img, iterations, out
