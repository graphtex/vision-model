import torch
import torchvision.transforms as T
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

def draw_bbox(raw_img, boxes, output_pil = True):
    boxes = box_convert(boxes, 'cxcywh','xyxy')
    pil_to_bytes = lambda I: T.functional.convert_image_dtype(T.ToTensor()(I), torch.uint8)
    raw_bytes = pil_to_bytes(raw_img)
    _, Y, X = raw_bytes.shape

    scale = torch.Tensor([X,Y,X,Y])
    boxes = torch.einsum('ij,j->ij',boxes,scale)
    boxes = boxes.to(torch.int)

    b = draw_bounding_boxes(raw_bytes, boxes)
    if output_pil:
        return T.ToPILImage()(b)
    else:
        return b



def get_clear_objs(im, model, thresh = 0.7, output_pil = True):
    transform = T.Compose([
        T.Resize(800),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(im).unsqueeze(0)
    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > thresh

    boxes = outputs['pred_boxes'][0,keep]
    return draw_bbox(im,boxes, output_pil, width=3)


def visualize_attention(img, model, thresh=0.85):
    # code to retrive attentions via hooks taken from the DETR official notebook: https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=hYVZjfGhYTEa

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights = [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    dec_attn_weights = dec_attn_weights[0]

    ps = outputs['pred_logits'].softmax(-1)[0,:,:-1]
    keep = ps.max(-1).values > thresh
    
    boxes = outputs['pred_boxes'][0,keep]
    boxes = box_convert(boxes, 'cxcywh', 'xyxy')
    img_bytes = T.functional.convert_image_dtype(img[0], torch.uint8)
    _,Y,X = img_bytes.shape
    scale = torch.Tensor([X,Y,X,Y])
    boxes = torch.einsum('ij,j->ij',boxes,scale)

    h, w = conv_features['0'].tensors.shape[-2:]

    N = 5
    fig, axs = plt.subplots(ncols=N, nrows=2,sharex='row', sharey='row', figsize=(3*N,6))
    imgs = []
    for i, idx, box in zip(range(N), keep.nonzero(), boxes):
        b = draw_bounding_boxes(img_bytes, box.unsqueeze(0), width=10, colors='red')
        imgs.append(b)
        axs[0,i].imshow(T.ToPILImage()(b))

        axs[1,i].imshow(dec_attn_weights[0, idx].view(h, w).detach())

        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
    
    plt.tight_layout()
        

    return fig