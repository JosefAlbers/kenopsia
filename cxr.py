from google.colab import drive
drive.mount('/content/gdrive')

# !pip install -qqq transformers==4.31.0 torchvision peft

import os, requests, torch, transformers, warnings
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from io import BytesIO
from torchvision.utils import make_grid
from urllib.request import urlopen, Request

ckpt_name = 'Adorg/cm231104'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_decoder = transformers.AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(device)
encoder_decoder.eval()
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(ckpt_name)
image_processor = transformers.AutoFeatureExtractor.from_pretrained(ckpt_name)

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

def get_images_for_one_patient_from_fns(fns_for_one_pt):
    images_for_one_pt = [test_transforms(Image.open(fn).convert('RGB')) for fn in fns_for_one_pt]
    study_for_one_pt = torch.stack(images_for_one_pt, dim=0)
    plt.imshow(make_grid(study_for_one_pt, normalize=True).permute(1, 2, 0))
    images = torch.nn.utils.rnn.pad_sequence([study_for_one_pt], batch_first=True, padding_value=0.0)
    return images

def get_images_for_one_patient_from_urls(urls_for_one_pt):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    images_for_one_pt = [test_transforms(Image.open(BytesIO(urlopen(Request(url, headers=headers)).read())).convert('RGB')) for url in urls_for_one_pt]
    study_for_one_pt = torch.stack(images_for_one_pt, dim=0)
    plt.imshow(make_grid(study_for_one_pt, normalize=True).permute(1, 2, 0))
    images = torch.nn.utils.rnn.pad_sequence([study_for_one_pt], batch_first=True, padding_value=0.0)
    return images

def get_prompt_from_previous_study(previous_findings, previous_impression):
    prompt = encoder_decoder.tokenize_prompt(
        previous_findings,
        previous_impression,
        tokenizer,
        256,
        add_bos_token_id=True,
    )
    return prompt

def generate_report(images, prompt):
    outputs = encoder_decoder.generate(
        pixel_values=images.to(device),
        decoder_input_ids=prompt['input_ids'],
        special_token_ids=[
            tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index('[PMT-SEP]')
            ],
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
        ],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        use_cache=True,
        max_length=256 + prompt['input_ids'].shape[1],
        num_beams=4,
    )

    if torch.all(outputs.sequences[:, 0] == 1):
        outputs.sequences = outputs.sequences[:, 1:]

    _, findings, impression = encoder_decoder.split_and_decode_sections(
        outputs.sequences,
        [tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id],
        tokenizer
    )

    for i, j in zip(findings, impression):
        print(f'Findings: {i}\nImpression: {j}\n')
    return findings, impression

def go(folder='Genius'):
    dir_path = f'/content/gdrive/My Drive/{folder}'
    fns = [f'{dir_path}/{fn}' for fn in os.listdir(dir_path)]
    images = get_images_for_one_patient_from_fns(fns)

    prompt = get_prompt_from_previous_study([None], [None])
    findings, impression = generate_report(images, prompt)

    # prompt2 = get_prompt_from_previous_study(findings, impression)
    # findings2, impression2 = generate_report(images, prompt2)

    return findings, impression

go('fromCrop')