from PIL import Image
import cv2
import sys
import torch
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator


# セットアップ
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

ckpt_dir='./OFA-tiny'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids


model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)

generator = sequence_generator.SequenceGenerator(
    tokenizer=tokenizer,
    beam_size=5,
    max_len_b=16,
    min_len=0,
    no_repeat_ngram_size=3,
)


def generate_caption(image_path="./saved.jpg"):
    "画像からキャプションを生成する関数"
    img = Image.open(image_path)
    patch_img = patch_resize_transform(img).unsqueeze(0)
    
    data = {}
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}

    gen_output = generator.generate([model], data)
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

    sentence = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return sentence


# リアルタイム処理
camera_id = 0   # カメラ番号
delay = 1
window_name = 'frame'
image_path = "./saved.jpg"  # 画像を保存するパス

cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    cv2.imwrite(image_path, frame)
    
    sentence = generate_caption(image_path)
    print(sentence) # 生成したキャプションを出力

    cv2.imshow(window_name, frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):   # "Q"が押されるまで無限ループ
        break

cv2.destroyWindow(window_name)