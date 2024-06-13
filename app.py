import streamlit as st
import nibabel as nib
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image


def read_nii(file_path):
    nii_data = nib.load(file_path).get_fdata().transpose(2, 1, 0)
    return nii_data


def normalize_image(image):
    max_val = torch.max(image)
    image[image < 0] = 0
    return image / max_val


def preprocess_image(image, im_h, im_w):
    trans = albumentations.Compose([
        albumentations.Resize(im_h, im_w),
        ToTensorV2(transpose_mask=True),
    ])
    transformed = trans(image=image)
    image = transformed['image']
    image = normalize_image(image)
    return image.float()


def predict(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze(0)
    return prediction


def predicter(nii_file, model, device):
    nii_data = read_nii(nii_file)
    preprocessed_images = [preprocess_image(slice, im_h, im_w) for slice in nii_data]
    predictions = [predict(slice, model, device) for slice in preprocessed_images]

    areas = torch.tensor(list(map(lambda prediction: torch.sum(prediction > 0), predictions)))
    max_area_idx = torch.argmax(areas)

    return nii_data[max_area_idx], predictions[max_area_idx]


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model_path = "liver_best_model_b.pt"
im_h, im_w = 256, 256

model = torch.load(model_path, map_location=torch.device(device))
model = model.to(device)


def main():
    st.title("Segmentation of nii-files of liver CT scans")

    uploaded_file = st.file_uploader("Upload NII file", type=["nii", "nii.gz"])

    if uploaded_file is not None:
        with open("temp_nii_file.nii", "wb") as f:
            f.write(uploaded_file.getbuffer())

        analiz_data, mask_data = predicter("temp_nii_file.nii", model, device)

        analiz_img = Image.fromarray(np.stack((analiz_data,) * 3, axis=-1).astype(np.uint8))
        predicted_image = Image.fromarray((mask_data.cpu().numpy() * 255).astype(np.uint8))

        st.image(analiz_img, caption='Original Image Slice', use_column_width=True)
        st.image(predicted_image, caption='Predicted Segmentation Mask', use_column_width=True)


if __name__ == "__main__":
    main()