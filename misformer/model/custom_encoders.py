from lavila.utils import distributed as dist_utils
import torch

def text_encoder(model, tokenizer, label, rank):
    templates = ['{}']

    text = [tmpl.format(label) for tmpl in templates]

    text = tokenizer(text)
    text = text.cuda(rank, non_blocking=True)
    mask = None
    text = text.view(-1, 77).contiguous()

    with torch.no_grad():
        embeddings = dist_utils.get_model(model).encode_text(text)
    
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def video_encoder(model, tensor, rank):
    tensor = tensor.cuda(rank, non_blocking=True)

    with torch.no_grad():
        features = dist_utils.get_model(model).encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    
    return features


if __name__ == "__main__":
    video_encoder()
    
    



