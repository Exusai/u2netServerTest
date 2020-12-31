import flask
from flask import Flask, redirect, url_for, request, render_template, send_file, Response
import os
from PIL import Image
import numpy as np
import base64
import io 
from skimage import transform
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms.functional as TF
from flask_cors import CORS, cross_origin

#from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def encodeToBase64(image_name, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    #img_name = image_name.split(os.sep)[-1]
    image = image_name
    #width, height = 
    imo = im.resize((image.size), resample = Image.BILINEAR)

    pb_np = np.array(imo)

    #aaa = img_name.split(".")
    #bbb = aaa[0:-1]
    #imidx = bbb[0]
    #for i in range(1,len(bbb)):
    #    imidx = imidx + "." + bbb[i]

    buffer = io.BytesIO()
    imo.save(buffer,format = "png")
    imageBuffer = buffer.getvalue()                 
    encodedImage = base64.b64encode(imageBuffer)
    return encodedImage
    #imo.save(d_dir+imidx+'.png')

def encodeZMapPNG(originalImage, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    im = Image.fromarray(predict_np*255).convert('RGB')
    #img_name = originalImage.split(os.sep)[-1]
    image = originalImage
    #width, height = 
    imo = im.resize((image.size), resample = Image.BILINEAR)

    pb_np = np.array(imo)

    #aaa = img_name.split(".")
    #bbb = aaa[0:-1]
    #imidx = bbb[0]
    #for i in range(1,len(bbb)):
    #    imidx = imidx + "." + bbb[i]

    buffer = io.BytesIO()
    imo.save(buffer, format = "PNG")
    return buffer.getvalue()       
    #encodedImage = base64.b64encode(imageBuffer)
    #return encodedImage
    #imo.save(d_dir+imidx+'.png')

def quitBGAndEncodePNG(originalImage, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    im = Image.fromarray(predict_np*255).convert('RGB')
    #img_name = originalImage.split(os.sep)[-1]
    image = originalImage
    #width, height = 
    imo = im.resize((image.size), resample = Image.BILINEAR)

    pb_np = np.array(imo,dtype=np.float64)
    
    ### RECORTAR ###
    RESCALE = 255
    out_img = pb_np
    out_img /= RESCALE

    THRESHOLD = 0.5

    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    
    # convert the rbg image to an rgba image and set the zero values to transparent
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

    inp_img = np.array(image, dtype=np.float64)
    inp_img /= RESCALE
    
    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = Image.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')

    buffer = io.BytesIO()
    rem_back_scaled.save(buffer, format = "PNG")
    return buffer.getvalue() 

    #rem_back_scaled.save(name+'_background_removed.png')

####
def quitBGAndEncodeBase64(originalImage, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    im = Image.fromarray(predict_np*255).convert('RGB')
    #img_name = originalImage.split(os.sep)[-1]
    image = originalImage
    #width, height = 
    imo = im.resize((image.size), resample = Image.BILINEAR)

    pb_np = np.array(imo,dtype=np.float64)
    
    ### RECORTAR ###
    RESCALE = 255
    out_img = pb_np
    out_img /= RESCALE

    THRESHOLD = 0.5

    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0
    
    # convert the rbg image to an rgba image and set the zero values to transparent
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

    inp_img = np.array(image, dtype=np.float64)
    inp_img /= RESCALE
    
    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = Image.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')

    buffer = io.BytesIO()
    rem_back_scaled.save(buffer, format = "PNG")    
    imageBuffer = buffer.getvalue()                 
    encodedImage = base64.b64encode(imageBuffer)
    return encodedImage

    #rem_back_scaled.save(name+'_background_removed.png')


@app.route('/')
def index():
    return "Quitador de fondo"

@app.route('/process', methods=['POST'])
def process():
    print('Staring Process')
    success = '0'
    

    print('Image on request')
    image = request.form.to_dict()['image']
    image = base64.b64decode(image)
    bufferImage = io.BytesIO(image)
    

    #print(image)
    #payload = request.form.to_dict() #usado para leer más información envida en el método POST
    #modelConfig = payload['model']   #Lee la info de model del método post

    model_name='u2netp'# fixed as u2netp
    model_dir = os.path.join(os.getcwd(), model_name + '.pth') # path to u2netp pretrained weights
    print('Model dir loaded')

    # --------- 3. model define ---------
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu'))) #Quitar "map_location=torch.device('cpu')" si hay GPU DIsponible
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    print('Model definition complete')
    image = Image.open(bufferImage).convert('RGB')
    originalImage = image
    print('Starting inference')

    
    

    ### Image must be resized to a certain bound otherwise the model will produce foggy like results and it will be significantly slower
    ### 240 can be changed, 320 is used in the original umplementation of u2net
    print(image.size)
    w, h = image.size
    if h == w:
        new_h, new_w = 240,240
    elif h > w:
        new_h, new_w = 240*h/w,240
    else:
        new_h, new_w = 240,240*w/h

    image = image.resize((int(new_w),int(new_h)), Image.BICUBIC)
    image = (image/np.max(image))
    inputs_test = TF.to_tensor(image)
    inputs_test = inputs_test.type(torch.FloatTensor)
    #print(inputs_test.size())
    #print(inputs_test)
    inputs_test[0,:,:] = (inputs_test[0,:,:]-0.485)/0.229
    inputs_test[1,:,:] = (inputs_test[1,:,:]-0.456)/0.224
    inputs_test[2,:,:] = (inputs_test[2,:,:]-0.406)/0.225
    
    inputs_test.unsqueeze_(0)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
    print('Inference complete')

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    
    print('saving and encoding')
    encodedImage = quitBGAndEncodeBase64(originalImage,pred)
    #encodedImageZMap = encodeZMapPNG(originalImage,pred)
    #enodedImageNoBG = quitBGAndEncodePNG(originalImage, pred)
    
    success = '1'

    del d1,d2,d3,d4,d5,d6,d7
    print('saved, encoded, waiting to finish request')
    img_str = encodedImage

    return flask.jsonify({'msg': str(success), 'img': str(img_str) })
    #return Response(enodedImageNoBG, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
