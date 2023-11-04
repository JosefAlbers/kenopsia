from google.colab import drive
drive.mount('/content/gdrive')

# !pip install -qqq transformers==4.31.0 torchvision

import os, torch, transformers
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from io import BytesIO
from torchvision.utils import make_grid

ckpt_name = 'Adorg/mc231104'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_decoder = transformers.AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(device).eval()
image_processor = transformers.AutoFeatureExtractor.from_pretrained(ckpt_name)
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(ckpt_name)

test_transforms = transforms.Compose(
    [
        transforms.Resize(size=image_processor.size['shortest_edge']),
        transforms.CenterCrop(size=[
            image_processor.size['shortest_edge'],
            image_processor.size['shortest_edge'],
        ]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std,
        ),
    ]
)

def get_images_for_one_patient_from_fns(folder):
    dir_path = f'/content/gdrive/My Drive/{folder}'
    fns_for_one_pt = [f'{dir_path}/{fn}' for fn in os.listdir(dir_path)]
    images_for_one_pt = [test_transforms(Image.open(fn).convert('RGB')) for fn in fns_for_one_pt]
    study_for_one_pt = torch.stack(images_for_one_pt, dim=0)
    plt.imshow(make_grid(study_for_one_pt, normalize=True).permute(1, 2, 0))
    # images = torch.nn.utils.rnn.pad_sequence([study_for_one_pt], batch_first=True, padding_value=0.0)
    return study_for_one_pt

def get_captions(folder='Genius'):
    images = get_images_for_one_patient_from_fns(folder)
    outputs = encoder_decoder.generate(
        pixel_values=images.to(device),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        use_cache=True,
        max_length=256,
        num_beams=4,
        # num_return_sequences=4,
    )
    captions = [tokenizer.decode(i, skip_special_tokens=True) for i in outputs.sequences]
    for c in captions:
        print(c)
    return captions

get_captions('fromCrop')