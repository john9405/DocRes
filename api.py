import base64
import io
import os
import sys
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

import utils
from data.preprocess.crop_merge_image import stride_integral
from models import restormer_arch
from utils import convert_state_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append('./data/MBD/')
from data.MBD.infer import net1_net2_infer_single_im

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dewarp_prompt(img):
    mask = net1_net2_infer_single_im(img, 'data/MBD/checkpoint/mbd.pkl')
    base_coord = utils.getBasecoord(256,256)/256
    img[mask==0]=0
    mask = cv2.resize(mask,(256,256))/255
    return img,np.concatenate((base_coord,np.expand_dims(mask,-1)),-1)


def deshadow_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    bg_imgs = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        bg_imgs.append(bg_img)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    bg_imgs = cv2.merge(bg_imgs)
    bg_imgs = cv2.resize(bg_imgs,(w,h))
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    result_norm[result_norm==0]=1
    shadow_map = np.clip(img.astype(float)/result_norm.astype(float)*255,0,255).astype(np.uint8)
    shadow_map = cv2.resize(shadow_map,(w,h))
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_BGR2GRAY)
    shadow_map = cv2.cvtColor(shadow_map,cv2.COLOR_GRAY2BGR)
    # return shadow_map
    return bg_imgs


def deblur_prompt(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_GRAY2BGR)
    return high_frequency


def appearance_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm,(w,h))
    return result_norm


def binarization_promptv2(img):
    result,thresh = utils.SauvolaModBinarization(img)
    thresh = thresh.astype(np.uint8)
    result[result>155]=255
    result[result<=155]=0

    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    high_frequency = cv2.addWeighted(absX,0.5,absY,0.5,0)
    high_frequency = cv2.cvtColor(high_frequency,cv2.COLOR_BGR2GRAY)
    return np.concatenate((np.expand_dims(thresh,-1),np.expand_dims(high_frequency,-1),np.expand_dims(result,-1)),-1)


def dewarping(model,im_path):
    INPUT_SIZE=256
    im_org = cv2.imread(im_path)
    im_masked, prompt_org = dewarp_prompt(im_org.copy())

    h,w = im_masked.shape[:2]
    im_masked = im_masked.copy()
    im_masked = cv2.resize(im_masked,(INPUT_SIZE,INPUT_SIZE))
    im_masked = im_masked / 255.0
    im_masked = torch.from_numpy(im_masked.transpose(2,0,1)).unsqueeze(0)
    im_masked = im_masked.float().to(DEVICE)

    prompt = torch.from_numpy(prompt_org.transpose(2,0,1)).unsqueeze(0)
    prompt = prompt.float().to(DEVICE)

    in_im = torch.cat((im_masked,prompt),dim=1)

    # inference
    base_coord = utils.getBasecoord(INPUT_SIZE,INPUT_SIZE)/INPUT_SIZE
    model = model.float()
    with torch.no_grad():
        pred = model(in_im)
        pred = pred[0][:2].permute(1,2,0).cpu().numpy()
        pred = pred+base_coord
    ## smooth
    for i in range(15):
        pred = cv2.blur(pred,(3,3),borderType=cv2.BORDER_REPLICATE)
    pred = cv2.resize(pred,(w,h))*(w,h)
    pred = pred.astype(np.float32)
    out_im = cv2.remap(im_org,pred[:,:,0],pred[:,:,1],cv2.INTER_LINEAR)

    prompt_org = (prompt_org*255).astype(np.uint8)
    prompt_org = cv2.resize(prompt_org,im_org.shape[:2][::-1])

    return prompt_org[:,:,0],prompt_org[:,:,1],prompt_org[:,:,2],out_im


def appearance(model,im_path):
    MAX_SIZE=1600
    # obtain im and prompt
    im_org = cv2.imread(im_path)
    h,w = im_org.shape[:2]
    prompt = appearance_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)

    # constrain the max resolution
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))

    # normalize
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)

    # inference
    in_im = in_im.half().to(DEVICE)
    model = model.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0] = 1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def deshadowing(model,im_path):
    MAX_SIZE=1600
    # obtain im and prompt
    im_org = cv2.imread(im_path)
    h,w = im_org.shape[:2]
    prompt = deshadow_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)

    # constrain the max resolution
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im,8)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))

    # normalize
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)

    # inference
    in_im = in_im.half().to(DEVICE)
    model = model.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0]=1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def deblurring(model,im_path):
    # setup image
    im_org = cv2.imread(im_path)
    in_im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = deblur_prompt(in_im)
    in_im = np.concatenate((in_im,prompt),-1)
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)
    in_im = in_im.half().to(DEVICE)
    # inference
    model.to(DEVICE)
    model.eval()
    model = model.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        out_im = pred[padding_h:,padding_w:]

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def binarization(model,im_path):
    im_org = cv2.imread(im_path)
    im,padding_h,padding_w = stride_integral(im_org,8)
    prompt = binarization_promptv2(im)
    h,w = im.shape[:2]
    in_im = np.concatenate((im,prompt),-1)

    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0)
    in_im = in_im.to(DEVICE)
    model = model.half()
    in_im = in_im.half()
    with torch.no_grad():
        pred = model(in_im)
        pred = pred[:,:2,:,:]
        pred = torch.max(torch.softmax(pred,1),1)[1]
        pred = pred[0].cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        pred = cv2.resize(pred,(w,h))
        out_im = pred[padding_h:,padding_w:]

    return prompt[:,:,0],prompt[:,:,1],prompt[:,:,2],out_im


def model_init():
   # prepare model
    model = restormer_arch.Restormer(
        inp_channels=6,
        out_channels=3,
        dim = 48,
        num_blocks = [2,3,3,4],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        dual_pixel_task = True
    )

    if DEVICE.type == 'cpu':
        state = convert_state_dict(torch.load('./checkpoints/docres.pkl', map_location='cpu')['model_state'])
    else:
        state = convert_state_dict(torch.load('./checkpoints/docres.pkl', map_location='cuda:0')['model_state'])
    model.load_state_dict(state)

    model.eval()
    model = model.to(DEVICE)
    return model


def inference_one_im(model,im_path,task):
    if task=='dewarping':
        prompt1,prompt2,prompt3,restorted = dewarping(model,im_path)
    elif task=='deshadowing':
        prompt1,prompt2,prompt3,restorted = deshadowing(model,im_path)
    elif task=='appearance':
        prompt1,prompt2,prompt3,restorted = appearance(model,im_path)
    elif task=='deblurring':
        prompt1,prompt2,prompt3,restorted = deblurring(model,im_path)
    elif task=='binarization':
        prompt1,prompt2,prompt3,restorted = binarization(model,im_path)
    elif task=='end2end':
        prompt1,prompt2,prompt3,restorted = dewarping(model,im_path)
        cv2.imwrite('restorted/step1.jpg',restorted)
        prompt1,prompt2,prompt3,restorted = deshadowing(model,'restorted/step1.jpg')
        cv2.imwrite('restorted/step2.jpg',restorted)
        prompt1,prompt2,prompt3,restorted = appearance(model,'restorted/step2.jpg')
        os.remove('restorted/step1.jpg')
        os.remove('restorted/step2.jpg')

    return prompt1,prompt2,prompt3,restorted

def get_extname(im_data: bytes):
    img = Image.open(io.BytesIO(im_data))
    return img.format.lower()

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    model = model_init()
    print("Model loaded successfully.")
    yield
    # Perform any necessary cleanup here (if needed)
    model = None
    print("Model cleanup done.")

possible_tasks = ['dewarping','deshadowing','appearance','deblurring','binarization','end2end']
origins = ["*"]

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/inference/")
async def inference(img_data: str = Body(...),
                    task: str = Body("dewarping"),
                    save_dtsprompt: int = Body(0)):

    if task not in possible_tasks:
        return JSONResponse(
            content={"error": f"Invalid task. Possible tasks are: {', '.join(possible_tasks)}"},
            status_code=400
        )
    if img_data is None:
        return JSONResponse(content={"error": "No image data provided"}, status_code=400)

    req_id = None
    im_path = None
    save_path = None
    im_format = None
    prompt_files = []

    try:
        img_data = base64.b64decode(img_data)
        req_id = str(uuid.uuid4())
        ext_name = get_extname(img_data)
        im_path = os.path.join(BASE_DIR, 'distorted', f"{req_id}.{ext_name}")
        with open(im_path, 'wb') as f:
            f.write(img_data)
        img_data = None
    except Exception as e:
        return JSONResponse(content={"error": f"Invalid image data: {str(e)}"}, status_code=400)

    try:
        ## inference
        prompt1,prompt2,prompt3,restorted = inference_one_im(model, im_path, task)

        out_folder = './restorted/'
        im_name = os.path.split(im_path)[-1]
        im_format = '.'+im_name.split('.')[-1]
        save_path = os.path.join(out_folder, im_name.replace(im_format, '_' + task + im_format))
        cv2.imwrite(save_path, restorted)
        if save_dtsprompt:
            prompt1_path = save_path.replace(im_format, '_prompt1' + im_format)
            prompt2_path = save_path.replace(im_format, '_prompt2' + im_format)
            prompt3_path = save_path.replace(im_format, '_prompt3' + im_format)
            cv2.imwrite(prompt1_path, prompt1)
            cv2.imwrite(prompt2_path, prompt2)
            cv2.imwrite(prompt3_path, prompt3)
            prompt_files = [prompt1_path, prompt2_path, prompt3_path]

        res_data = {
            "result":"",
            "prompt1": "",
            "prompt2": "",
            "prompt3": ""
        }
        with open(save_path, 'rb') as f:
            res_data['result'] = base64.b64encode(f.read()).decode()
        os.remove(save_path)
        if save_dtsprompt:
            with open(prompt_files[0], 'rb') as f:
                res_data['prompt1'] = base64.b64encode(f.read()).decode()
            os.remove(prompt_files[0])

            with open(prompt_files[1], 'rb') as f:
                res_data['prompt2'] = base64.b64encode(f.read()).decode()
            os.remove(prompt_files[1])

            with open(prompt_files[2], 'rb') as f:
                res_data['prompt3'] = base64.b64encode(f.read()).decode()
            os.remove(prompt_files[2])

        return JSONResponse(content=res_data, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"Inference failed: {str(e)}"}, status_code=500)

    finally:
        # Clean up temporary files
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
        for prompt_file in prompt_files:
            if os.path.exists(prompt_file):
                os.remove(prompt_file)
        if im_path and os.path.exists(im_path):
            os.remove(im_path)


@app.get('/')
def index():
    page_data = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API 测试工具 - Dewarping</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }

        .upload-area:hover {
            border-color: #007AFF;
            background: #f8fbff;
        }

        .upload-area.dragover {
            border-color: #007AFF;
            background: #e8f4ff;
        }

        .upload-icon {
            font-size: 48px;
            color: #999;
            margin-bottom: 12px;
        }

        .upload-text {
            color: #666;
            font-size: 16px;
        }

        #fileInput {
            display: none;
        }

        .btn {
            background: #007AFF;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            width: 100%;
        }

        .btn:hover {
            background: #0051D5;
        }

        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .image-preview-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .image-box {
            flex: 1;
            min-width: 300px;
        }

        .image-box h3 {
            margin-bottom: 12px;
            color: #333;
            font-size: 18px;
        }

        .image-wrapper {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            background: #fafafa;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .image-wrapper img {
            max-width: 100%;
            max-height: 400px;
            object-fit: contain;
        }

        .placeholder {
            color: #999;
            font-size: 14px;
        }

        .info-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .time-display {
            font-size: 14px;
            color: #666;
        }

        .time-display .value {
            font-weight: bold;
            color: #007AFF;
            font-size: 18px;
        }

        .status {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }

        .status.loading {
            background: #fff3cd;
            color: #856404;
        }

        .status.success {
            background: #d4edda;
            color: #155724;
        }

        .status.error {
            background: #f8d7da;
            color: #721c24;
        }

        .options {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            align-items: center;
        }

        .options label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 14px;
        }

        .options input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }

        .options select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            background: white;
            cursor: pointer;
            min-width: 150px;
        }

        .options select:focus {
            outline: none;
            border-color: #007AFF;
        }

        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 12px;
            font-size: 14px;
        }

        .prompts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }

        .prompt-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }

        .prompt-item h4 {
            background: #f5f5f5;
            padding: 8px 12px;
            font-size: 14px;
            color: #666;
            border-bottom: 1px solid #e0e0e0;
        }

        .prompt-item img {
            width: 100%;
            height: 150px;
            object-fit: contain;
            background: #fafafa;
        }

        .action-bar {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }

        .btn-secondary {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
            flex: 1;
        }

        .btn-secondary:hover {
            background: #218838;
        }

        .btn-secondary:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .loading-spinner {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
            padding: 40px;
        }

        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007AFF;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            color: #666;
            font-size: 14px;
        }

        .error-icon {
            font-size: 48px;
            color: #dc3545;
        }

        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ 图像去畸变 API 测试工具</h1>

        <!-- 上传区域 -->
        <div class="card">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">点击选择图片或拖拽图片到此处</div>
            </div>
            <input type="file" id="fileInput" accept="image/*">

            <div class="options" style="margin-top: 20px;">
                <label>
                    处理类型:
                    <select id="taskType">
                        <option value="dewarping">dewarping (去畸变)</option>
                        <option value="deshadowing">deshadowing (去阴影)</option>
                        <option value="appearance">appearance (外观增强)</option>
                        <option value="deblurring">deblurring (去模糊)</option>
                        <option value="binarization">binarization (二值化)</option>
                        <option value="end2end">end2end (端到端)</option>
                    </select>
                </label>
                <label>
                    <input type="checkbox" id="savePrompt" value="1">
                    保存中间过程图 (save_dtsprompt)
                </label>
            </div>
        </div>

        <!-- 图片对比区域 -->
        <div class="card" id="resultCard" style="display: none;">
            <div class="image-preview-container">
                <div class="image-box">
                    <h3>📥 原始图片</h3>
                    <div class="image-wrapper">
                        <img id="originalImage" src="" alt="原始图片">
                    </div>
                </div>
                <div class="image-box">
                    <h3>📤 处理结果</h3>
                    <div class="image-wrapper">
                        <div id="loadingState" class="loading-spinner hidden">
                            <div class="spinner" id="spinnerIcon"></div>
                            <div class="error-icon hidden" id="errorIcon">✕</div>
                            <div class="loading-text" id="loadingText">正在处理中...</div>
                        </div>
                        <img id="resultImage" src="" alt="处理结果" class="hidden">
                    </div>
                </div>
            </div>

            <div class="info-bar">
                <div class="time-display">
                    处理时长: <span class="value" id="processingTime">-</span> 秒
                </div>
                <div class="status" id="status"></div>
            </div>

            <div class="action-bar">
                <button class="btn-secondary" id="retryBtn" disabled>🔄 重新处理</button>
            </div>

            <div id="promptsContainer" class="prompts-grid" style="display: none;"></div>
            <div id="errorMessage"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const originalImage = document.getElementById('originalImage');
        const resultImage = document.getElementById('resultImage');
        const resultCard = document.getElementById('resultCard');
        const processingTime = document.getElementById('processingTime');
        const statusEl = document.getElementById('status');
        const savePromptCheckbox = document.getElementById('savePrompt');
        const taskTypeSelect = document.getElementById('taskType');
        const promptsContainer = document.getElementById('promptsContainer');
        const errorMessage = document.getElementById('errorMessage');
        const retryBtn = document.getElementById('retryBtn');
        const loadingState = document.getElementById('loadingState');
        const spinnerIcon = document.getElementById('spinnerIcon');
        const errorIcon = document.getElementById('errorIcon');
        const loadingText = document.getElementById('loadingText');

        let currentFile = null;
        let currentBase64 = null;

        // 点击上传
        uploadArea.addEventListener('click', () => fileInput.click());

        // 重试按钮
        retryBtn.addEventListener('click', () => {
            if (currentBase64) {
                promptsContainer.style.display = 'none';
                promptsContainer.innerHTML = '';
                errorMessage.innerHTML = '';
                resultImage.src = '';
                loadingState.classList.add('hidden');
                resultImage.classList.add('hidden');
                spinnerIcon.classList.remove('hidden');
                errorIcon.classList.add('hidden');
                loadingText.textContent = '正在处理中...';
                processingTime.textContent = '-';
                statusEl.className = 'status';
                statusEl.textContent = '';
                callAPI(currentBase64);
            }
        });

        // 文件选择
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // 处理文件
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('请选择图片文件');
                return;
            }

            currentFile = file;

            // 显示原始图片
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImage.src = e.target.result;
                currentBase64 = e.target.result.split(',')[1];
                resultCard.style.display = 'block';
                promptsContainer.style.display = 'none';
                promptsContainer.innerHTML = '';
                errorMessage.innerHTML = '';
                resultImage.src = '';
                resultImage.classList.add('hidden');
                loadingState.classList.add('hidden');
                spinnerIcon.classList.remove('hidden');
                errorIcon.classList.add('hidden');
                loadingText.textContent = '正在处理中...';
                processingTime.textContent = '-';
                statusEl.className = 'status';
                statusEl.textContent = '';
                retryBtn.disabled = true;

                // 自动调用API
                callAPI(currentBase64);
            };
            reader.readAsDataURL(file);
        }

        // 调用API
        async function callAPI(base64Data) {
            statusEl.className = 'status loading';
            statusEl.textContent = '⏳ 处理中...';
            errorMessage.innerHTML = '';
            retryBtn.disabled = true;

            // 显示加载状态
            loadingState.classList.remove('hidden');
            resultImage.classList.add('hidden');
            spinnerIcon.classList.remove('hidden');
            errorIcon.classList.add('hidden');
            loadingText.textContent = '正在处理中...';

            const save_dtsprompt = savePromptCheckbox.checked ? 1 : 0;
            const task = taskTypeSelect.value;

            const startTime = performance.now();

            try {
                const response = await fetch('/api/inference/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        img_data: base64Data,
                        task: task,
                        save_dtsprompt: save_dtsprompt
                    })
                });

                const endTime = performance.now();
                const duration = ((endTime - startTime) / 1000).toFixed(2);
                processingTime.textContent = duration;

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                // 显示结果图片
                if (data.result) {
                    resultImage.src = 'data:image/png;base64,' + data.result;
                    loadingState.classList.add('hidden');
                    resultImage.classList.remove('hidden');
                    statusEl.className = 'status success';
                    statusEl.textContent = '✅ 处理成功';
                }

                // 显示中间过程图
                if (save_dtsprompt === 1) {
                    promptsContainer.style.display = 'grid';
                    promptsContainer.innerHTML = '';

                    const promptKeys = ['prompt1', 'prompt2', 'prompt3'];
                    const promptNames = ['中间过程 1', '中间过程 2', '中间过程 3'];

                    promptKeys.forEach((key, index) => {
                        if (data[key]) {
                            const item = document.createElement('div');
                            item.className = 'prompt-item';
                            item.innerHTML = `
                                <h4>${promptNames[index]}</h4>
                                <img src="data:image/jpg;base64,${data[key]}" alt="${promptNames[index]}">
                            `;
                            promptsContainer.appendChild(item);
                        }
                    });
                }

            } catch (error) {
                console.error('API Error:', error);
                statusEl.className = 'status error';
                statusEl.textContent = '❌ 处理失败';
                errorMessage.innerHTML = `<div class="error-message"><strong>错误信息:</strong> ${error.message}<br><br>请确保:<br>1. API 服务器正在运行 (http://127.0.0.1:8000)<br>2. CORS 配置正确</div>`;

                // 显示错误图标
                spinnerIcon.classList.add('hidden');
                errorIcon.classList.remove('hidden');
                loadingText.textContent = '处理失败';
            }

            // 无论成功或失败，都启用重试按钮
            retryBtn.disabled = false;
        }
    </script>
</body>
</html>
"""
    return Response(page_data, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
