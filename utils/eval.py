from PIL import Image, ImageDraw
from torchvision import transforms
import torch
import piq


def get_metric_fn(args):
    img_size = args.data.resolution
    mask = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.pieslice([0, 0, img_size, img_size], 0, 360, fill=255)
    toTensor = transforms.ToTensor()
    mask = toTensor(mask)[0]
    def metric_fn(predictions, targets, mask_roi=False, hist_norm=False):
        with torch.no_grad():
            if hist_norm:
                pred_hist = torch.histc(predictions, bins=255)
                targ_hist = torch.histc(targets, bins=255)

                peak_pred1 = torch.argmax(pred_hist[:75]) / 255.
                peak_pred2 = (torch.argmax(pred_hist[75:]) + 75) / 255.
                peak_targ1 = torch.argmax(targ_hist[:75]) / 255.
                peak_targ2 = (torch.argmax(targ_hist[75:]) + 75) / 255.

                predictions = torch.clamp((predictions - peak_pred1) / (peak_pred2 - peak_pred1), min=0)
                targets = torch.clamp((targets - peak_targ1) / (peak_targ2 - peak_targ1), min=0)

                predictions = torch.clamp(predictions, max=torch.max(targets).item(), min=0)
                predictions /= torch.max(targets)
                targets /= torch.max(targets)

            # Mask Region of Interest
            if mask_roi:
                predictions = predictions * mask
                targets = targets * mask

            return (piq.psnr(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item(),
                    piq.ssim(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item())
    return metric_fn