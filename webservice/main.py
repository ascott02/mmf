import io
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
os.chdir(parent)

# for web.py
import web
import re
import config
import base64
import datetime

urls = (
    '/', 'index',
    '/api', 'api',
    '/login', 'login',
    '/upload', 'upload',
    '/batch', 'batch',
)

import logging
log = logging.getLogger(config.log_file)
if not len(log.handlers):
    log.setLevel(logging.INFO)
    loghandler = logging.FileHandler(config.log_file)
    log.addHandler(loghandler)

import json
import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO
from argparse import Namespace

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

from mmf.datasets.processors.processors import VocabProcessor, CaptionProcessor
from mmf.models.butd import BUTD
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.env import setup_imports
from mmf.utils.configuration import Configuration


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False

# from mmf.utils.env
setup_imports()

model_yaml = "content/model_data/butd.yaml"
model_pth = 'content/model_data/butd.pth'

# model_pth = "/home/907308160/.cache/torch/mmf/data/models/butd.defaults/model.pth"
# model_yaml = "/home/907308160/.cache/torch/mmf/data/models/butd.defaults/config.yaml"

class PythiaDemo:
  TARGET_IMAGE_SIZE = [448, 448]
  CHANNEL_MEAN = [0.485, 0.456, 0.406]
  CHANNEL_STD = [0.229, 0.224, 0.225]
  def __init__(self):
    self._init_processors()
    self.pythia_model = self._build_pythia_model()
    self.detection_model = self._build_detection_model()

  def _init_processors(self):
    with open(model_yaml) as f:
      # config = yaml.load(f)
      config = yaml.safe_load(f)

    args = Namespace()
    args.opts = [
        "config="+model_yaml,
        "datasets=coco",
        "model=butd",
        "evaluation.predict=True"
    ]
    args.config_override = None


    configuration = Configuration(args=args)
    config = self.config = configuration.config

    # Remove warning
    config.training_parameters.evalai_inference = True
    registry.register("config", config)

    captioning_config = config.task_attributes.captioning.dataset_attributes.coco
    text_processor_config = captioning_config.processors.text_processor
    caption_processor_config = captioning_config.processors.caption_processor

    self.text_processor = VocabProcessor(text_processor_config.params)
    self.caption_processor = CaptionProcessor(caption_processor_config.params)

    registry.register("coco_text_processor", self.text_processor)
    registry.register("coco_caption_processor", self.caption_processor)

  def _build_pythia_model(self):
    state_dict = torch.load(model_pth)
    model_config = self.config.model_attributes.butd
    model_config.model_data_dir = "content/"
    model = BUTD(model_config)
    model.build()
    # model.init_losses_and_metrics()

    if list(state_dict.keys())[0].startswith('module') and \
       not hasattr(model, 'module'):
      state_dict = self._multi_gpu_state_to_single(state_dict)

    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()

    return model

  def _multi_gpu_state_to_single(self, state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd

  def predict(self, url):
    with torch.no_grad():
      detectron_features_list = self.get_detectron_features(url)

      sample_list = [] 

      for i, detectron_features in enumerate(detectron_features_list):
          sample = Sample()
          sample.dataset_name = "coco"
          sample.dataset_type = "test"
          sample.image_feature_0 = detectron_features
          sample.answers = torch.zeros((1,1), dtype=torch.long)
          sample_list.append(sample)

      # sample_list = SampleList([sample])
      sample_list = SampleList(sample_list)
      sample_list = sample_list.to("cuda")
      tokens = self.pythia_model(sample_list)["captions"]

    gc.collect()
    torch.cuda.empty_cache()

    return tokens


  def _build_detection_model(self):

      cfg.merge_from_file('content/model_data/detectron_model.yaml')
      cfg.freeze()

      model = build_detection_model(cfg)
      checkpoint = torch.load('content/model_data/detectron_model.pth',
                              map_location=torch.device("cpu"))

      load_state_dict(model, checkpoint.pop("model"))

      model.to("cuda")
      model.eval()
      return model

  def get_actual_image(self, image_path):
      if type(image_path) == type(str()) and image_path.startswith('http'):
          path = requests.get(image_path, stream=True)
          try:
              img = Image.open(path).convert('RGB') 
          except:
              stream = BytesIO(path.content) 
              img = Image.open(stream).convert('RGB')

      elif type(image_path) == type(str()) and image_path.startswith('/'):
          img = Image.open(image_path).convert('RGB')
      else:
          # print("DEBUG type(image_path):", type(image_path))
          # print("DEBUG image_path:", image_path[:20])
          stream = BytesIO(image_path) 
          img = Image.open(stream).convert('RGB')

      return img

  def _image_transform(self, image_path):
      img = self.get_actual_image(image_path)
      # img = Image.open(image_path)

      im = np.array(img).astype(np.float32)
      im = im[:, :, ::-1]
      im -= np.array([102.9801, 115.9465, 122.7717])
      im_shape = im.shape
      im_height = im_shape[0]
      im_width = im_shape[1]
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
      im_scale = float(800) / float(im_size_min)
      # Prevent the biggest axis from being more than max_size
      if np.round(im_scale * im_size_max) > 1333:
           im_scale = float(1333) / float(im_size_max)
      im = cv2.resize(
           im,
           None,
           None,
           fx=im_scale,
           fy=im_scale,
           interpolation=cv2.INTER_LINEAR
       )
      img = torch.from_numpy(im).permute(2, 0, 1)
      im_info = {"width": im_width, "height": im_height}
      return img, im_scale, im_info


  def _process_feature_extraction(self, output,
                                 im_scales,
                                 feat_name='fc6',
                                 conf_thresh=0.2):
      batch_size = len(output[0]["proposals"])
      n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
      score_list = output[0]["scores"].split(n_boxes_per_image)
      score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
      feats = output[0][feat_name].split(n_boxes_per_image)
      cur_device = score_list[0].device

      feat_list = []

      for i in range(batch_size):
          dets = output[0]["proposals"][i].bbox / im_scales[i]
          scores = score_list[i]

          max_conf = torch.zeros((scores.shape[0])).to(cur_device)

          for cls_ind in range(1, scores.shape[1]):
              cls_scores = scores[:, cls_ind]
              keep = nms(dets, cls_scores, 0.5)
              max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                           cls_scores[keep],
                                           max_conf[keep])

          keep_boxes = torch.argsort(max_conf, descending=True)[:100]
          feat_list.append(feats[i][keep_boxes])
      return feat_list

  def masked_unk_softmax(self, x, dim, mask_idx):
      x1 = F.softmax(x, dim=dim)
      x1[:, mask_idx] = 0
      x1_sum = torch.sum(x1, dim=1, keepdim=True)
      y = x1 / x1_sum
      return y

  def get_detectron_features(self, image_paths):
      img_tensor, im_scales, im_infos = [], [], []

      for image_path in image_paths:
          im, im_scale, im_info = self._image_transform(image_path)
          img_tensor.append(im)
          im_scales.append(im_scale)
          im_infos.append(im_info)
       
      current_img_list = to_image_list(img_tensor, size_divisible=32)
      current_img_list = current_img_list.to('cuda')
      with torch.no_grad():
          output = self.detection_model(current_img_list)

      feat_list = self._process_feature_extraction(output, im_scales,
                                                  'fc6', 0.2)
      # return feat_list[0]
      return feat_list


demo = PythiaDemo()



def get_caption(image_batch):

    for i,image in enumerate(image_batch):
        if uri_validator(image):
            try:
                response = requests.get(image, stream=True)
            except Exception as e:
                print(f"Exception occurred retrieving image {image}: {str(e)}")
                continue
            if response.statuscode == 200:
                # set decod content value to True, otherwise the downloaded image file's size will be zero
                response.raw.decode_content = True
                image_batch[i] = response.raw
            else:
                print(f"Image {image} could not be retrieved. Status code {response.status_code} returned.")
                continue
    
    tokens = [demo.predict(image_batch)]
    captions = []
    for token in tokens:
       captions.append(demo.caption_processor(token.tolist()[0])['caption'])

    return captions



class index:
    def GET(self, *args):
        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return """<html><head></head><body>
This form takes an image upload returns caption.<br/><br/>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
<br/><br/>
<input type="submit" />
</form>
</body></html>"""
        else:
            raise web.seeother('/login')

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['img_file'].filename)    # This is the filename

        caption = get_caption([x['img_file'].value])

        # tokens = demo.predict(x['img_file'].value)
        # caption = demo.caption_processor(tokens.tolist()[0])['caption']

        data_uri = base64.b64encode(x['img_file'].value)
        img_tag = '<img src="data:image/jpeg;base64,{0}">'.format(data_uri.decode())

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /index img_file: {x['img_file'].filename}")

        page = """<html><head></head><body>
This form takes an image upload and caption and returns an IICR rating.<br/><br/>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
<br/><br/>
<input type="submit" />
</form>""" + img_tag + """<br/>Caption: """ + caption[0] + """<br/>
</body></html>"""

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return page
        else:
            raise web.seeother('/login')


class api:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['img_url'])                # This is the URL to the image

        if not x['img_url']:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."
    

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /api token: {x['token']}, img_url: {x['img_url']}")
        caption = get_caption([x['img_url']])
        return caption


class batch:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['json_payload'])           # This is the URL to the image

        if not x['json_payload']:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /batch token: {x['token']}, json_payload: {x['json_payload']}")
        json_dict = json.loads(x['json_payload'])
        captions = {}
        for image in json_dict['images']:
            # caption = get_caption(json_dict['images'])
            captions[image] = get_caption([image])
        # return [x['caption'] for x in captions]
        return json.dumps(captions)
        # captions = get_caption(json_dict['images'])
        # return captions


class upload:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        # web.debug(x['img_data'].filename)      # This is the filename

        if not x['img_data']:
            return "No file."

        if not x['token']:
            return "No token."
        token = x['token'].decode()

        if not token in config.tokens:
            return "Not in tokens."
    
        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /upload token: {token}")
        caption = get_caption([x['img_data']])
        return caption


class login:

    def GET(self):
        auth = web.ctx.env.get('HTTP_AUTHORIZATION')
        authreq = False
        if auth is None:
            authreq = True
        else:
            auth = re.sub('^Basic ','',auth)
            print("DEBUG auth:", auth)
            # username,password = base64.decodestring(auth).split(':')
            username,password = base64.b64decode(auth).decode().split(':')
            if (username,password) in config.allowed:
                raise web.seeother('/')
            else:
                authreq = True
        if authreq:
            web.header('WWW-Authenticate','Basic realm="Auth example"')
            web.ctx.status = '401 Unauthorized'
            return

app = web.application(urls, globals(), autoreload=False)

if __name__ == "__main__":
    app.run()
