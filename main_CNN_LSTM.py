import cv2
import sys
import pickle
import torch
from torchvision import transforms
from utils import load_image, Vocabulary, Encoder, Decoder


# セットアップ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_path ='models/encoder-5-3000.pkl'
decoder_path ='models/decoder-5-3000.pkl'
vocab_path ='models/vocab.pkl'

embed_size=256
hidden_size=512
num_layers=1

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

encoder = Encoder(embed_size).eval()
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))


def generate_caption(image_path="./saved.jpg"):
    "画像からキャプションを生成する関数"
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)
    predicted_ids = decoder.predict(feature)
    predicted_ids = predicted_ids[0].cpu().numpy() 

    predicted_caption = []
    for word_id in predicted_ids:
        word = vocab.idx2word[word_id]
        predicted_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(predicted_caption[1:-2])
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