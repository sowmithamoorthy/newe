from django.shortcuts import render, redirect
from django.conf import settings
from .forms import VideoUploadForm
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import face_recognition
from torch import nn
import time
import shutil
from PIL import Image as pImage
import glob

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))



class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)


    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image


def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image


def predict(model, img, path='./', video_file_name=""):
    fmap, logits = model(img.to(device))
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]


def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "model_87_acc_150_frames_final_data.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")

    return final_model


ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        request.session.pop('file_name', None)
        request.session.pop('preprocessed_images', None)
        request.session.pop('faces_cropped_images', None)
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})

            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed")
                return render(request, index_template_name, {"form": video_upload_form})

            saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
            upload_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            if not settings.DEBUG:
                upload_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', 'app', 'uploaded_videos', saved_video_file)

            with open(upload_path, 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)

            request.session['file_name'] = upload_path
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})


def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:index")
        video_file = request.session['file_name']
        sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]

        # Load validation dataset
        video_dataset = ValidationDataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        model = Model(2).to(device)
        model_path = os.path.join(settings.PROJECT_DIR, 'models', 'ch.pt')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Process and predict
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None)) if ret]
        cap.release()

        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_name = f"{video_file_name_only}preprocessed{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            pImage.fromarray(rgb_frame, 'RGB').save(image_path)
            preprocessed_images.append(image_name)

            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                frame_face = frame[top - 40:bottom + 40, left - 40:right + 40]
                img_face_rgb = pImage.fromarray(cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB), 'RGB')
                image_name = f"{video_file_name_only}cropped_faces{i+1}.png"
                image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
                img_face_rgb.save(image_path)
                faces_cropped_images.append(image_name)

        if not faces_cropped_images:
            return render(request, predict_template_name, {"no_faces": True})

        try:
            prediction = predict(model, video_dataset[0], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            output = "REAL" if prediction[0] == 1 else "FAKE"
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'output': output,
                'confidence': confidence
            }
            return render(request, predict_template_name, context)
        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')



def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')
