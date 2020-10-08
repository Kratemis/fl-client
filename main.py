import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import json
import boto3
from botocore.exceptions import NoCredentialsError
import time
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument('--s3-client-models-folder', help='S3 folder for client models', required=True)
parser.add_argument('--s3-main-models-folder', help='S3 folder for main models', required=True)
parser.add_argument('--local-dataset-folder', help='Local folder for dataset', required=True)
parser.add_argument('--local-client-models-folder', help='Local folder for client models', required=True)
parser.add_argument('--local-main-model-folder', help='Local folder for client models', required=True)
parser.add_argument('--config-file', help='Configuration file with ML parameters', required=True)
parser.add_argument('--job-id', help='Unique Job ID', required=True)
parser.add_argument('--clients-bucket', help='Bucket name for client models', required=True)
parser.add_argument('--main-bucket', help='Bucket name for main models', required=True)
parser.add_argument('--s3-access-key', help='Credentials for AWS', required=False)
parser.add_argument('--s3-secret-key', help='Credentials for AWS', required=False)
parser.add_argument('-d', '--debug', help="Debug mode for the script")

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


def load_config():
    logging.info('Loading config')
    return json.loads(str(args.config_file))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def upload_to_aws(local_file, bucket, s3_file):
    logging.info("Uploading to S3 bucket")

    s3 = boto3.client('s3', aws_access_key_id=args.s3_access_key,
                      aws_secret_access_key=args.s3_secret_key)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        logging.info("Upload Successful")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False


def download_from_aws(bucket, remote_path, local_path):
    logging.info("Downloading from S3 bucket")

    s3 = boto3.client('s3', aws_access_key_id=args.s3_access_key,
                      aws_secret_access_key=args.s3_secret_key)

    try:
        logging.info("Bucket: " + bucket)
        logging.info("Remote Path: " + remote_path)
        logging.info("Local Path: " + local_path)
        s3.download_file(bucket, remote_path, local_path)
        logging.info("Download Successful")
        return True
    except FileNotFoundError:
        logging.error("The file was not found")
        return False
    except NoCredentialsError:
        logging.error("Credentials not available")
        return False


config = load_config()

device = torch.device("cuda:0" if config['use_cuda'] else "cpu")

logging.info("CUDA: ")
logging.info(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=args.local_dataset_folder, train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(config['batch_size']),
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

S3_MAIN_MODEL_PATH = args.s3_main_models_folder + '/main_model.pt'
LOCAL_MAIN_MODEL_PATH = args.local_main_model_folder + '/main_model.pt'
download_from_aws(args.main_bucket, S3_MAIN_MODEL_PATH, LOCAL_MAIN_MODEL_PATH)

net = torch.load(LOCAL_MAIN_MODEL_PATH)
net.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=float(config['lr']), momentum=float(config['momentum']))

for epoch in range(int(config['epochs'])):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if config['use_cuda']:
            inputs, labels = data[0].to(device)
        else:
            inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            logging.info('[%d, %5d] loss: %.3f' %
                         (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

logging.info('Finished Training')

logging.info('Saving model...')

MODEL = str(int(time.time())) + '_model.pt'
MODEL_PATH = args.local_client_models_folder + '/' + MODEL

torch.save(net, MODEL_PATH, _use_new_zipfile_serialization=False)
uploaded = upload_to_aws(MODEL_PATH, args.clients_bucket, args.s3_client_models_folder + '/' + MODEL)
