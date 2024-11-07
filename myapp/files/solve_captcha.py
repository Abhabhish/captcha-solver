import torch
from . import config
from . import dataset
from .model import CaptchaModel
from sklearn import preprocessing
import pickle
import numpy as np
import os
from django.conf import settings


# Load the saved label encoder
def load_label_encoder():
    with open(f"{settings.BASE_DIR}/myapp/files/label_encoder.pkl", "rb") as f:
        lbl_enc = pickle.load(f)
    return lbl_enc

# Load and prepare your saved model
def load_model(model_path, num_chars):
    model = CaptchaModel(num_chars=num_chars)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.to(config.DEVICE)
    model.eval()
    return model

# Decode predictions to characters
def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = preds.argmax(dim=2).detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j,:]:
            k = k - 1
            if k == -1:
                temp.append("°")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds

# Make predictions on a new CAPTCHA image
def predict_captcha(model, image_path, lbl_enc):
    test_dataset = dataset.ClassificationDataset(
        image_paths=[image_path],
        targets=[[0, 0, 0, 0, 0]],  # Placeholder for target
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    
    with torch.no_grad():
        for data in test_loader:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)
            preds, _ = model(**data)
            return decode_predictions(preds, lbl_enc)[0]

# Main function to solve a new CAPTCHA
# import os
# if __name__ == "__main__":
#     model_path = "./captcha_model.pth"  # Path to your saved model
#     test_dir = "../input/test"
#     lbl_enc = load_label_encoder()
#     model = load_model(model_path, num_chars=len(lbl_enc.classes_))
#     for img in os.listdir(test_dir):
#         img_path = os.path.join(test_dir,img)
#         prediction = predict_captcha(model, img_path, lbl_enc).replace('°','')
#         print(f"{img}: {prediction}")

def solve_captcha(img_path):
    model_path = f"{settings.BASE_DIR}/myapp/files/captcha_model.pth"
    lbl_enc = load_label_encoder()
    model = load_model(model_path, num_chars=len(lbl_enc.classes_))
    prediction = predict_captcha(model, img_path, lbl_enc).replace('°','')
    return prediction