import argparse
import logging
import sagemaker_containers
import requests

import os
import io
import glob
import time

from fastai.vision import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

def _train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading dataset")
    print(f'Batch size: {args.batch_size}')
    path = Path(args.data_dir)
    print(f'Data path is: {path}')
 
    # get the pattern to select the training/validation data
    np.random.seed(3)
    print('Creating DataBunch object')
    data = ImageDataBunch.from_folder(path, 
                                       ds_tfms=get_transforms(), 
                                       size=args.image_size, 
                                       bs=args.batch_size).normalize(imagenet_stats)
    # create the CNN model
    print('Create CNN model')
    print(f'Model architecture is {args.model_arch}')
    arch = getattr(models, args.model_arch)
    print("Creating conv net")    
    learn = cnn_learner(data, arch,pretrained=True,model_dir='/tmp',metrics=error_rate)
    print('Fit for 4 cycles')
    learn.fit_one_cycle(4)
    print('Finished Training')
    
    logger.info("Saving the model.")
    model_path = Path(args.model_dir)
    print(f'Export data object')
    data.export(model_path/'export.pkl')
    # create empty models dir
    os.mkdir(model_path/'models')
    print(f'Saving model weights')
    return learn.save(model_path/f'{args.model_arch}')

def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    print('Creating DataBunch object')
    empty_data = ImageDataBunch.load_empty(path)
    arch_name = os.path.splitext(os.path.split(glob.glob(f'{model_dir}/*.pth')[0])[1])[0]
    print(f'Model architecture is: {arch_name}')
    arch = getattr(models, arch_name)
    # Had to add model_dir to a path that I knew was writable as the models dir apparently gets mounted as RO, at least initally
    learn = cnn_learner(empty_data, arch,pretrained=False,model_dir='/tmp').load(path/f'{arch_name}')
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        img = open_image(io.BytesIO(request_body))
        return img
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        img = open_image(io.BytesIO(img_request.content))
        return img        
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = predict_values[predict_idx.item()].item()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        output = json.dumps(prediction)
        return output, accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

     # fast.ai specific parameters
    parser.add_argument('--image-size', type=int, default=224, metavar='IS',
                        help='image size (default: 224)')
    parser.add_argument('--model-arch', type=str, default='resnet50', metavar='MA',
                        help='model arch (default: resnet50)')   
    
    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    _train(parser.parse_args())
