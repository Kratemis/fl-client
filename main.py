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
import logging
import os


def load_config():
    return json.loads(str(os.environ['CONFIG']))


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

    s3 = boto3.client('s3', aws_access_key_id=config['secrets']['s3_access_key'],
                      aws_secret_access_key=config['secrets']['s3_secret_key'],
                      aws_session_token=config['secrets']['s3_session_token'])

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

    s3 = boto3.client('s3', aws_access_key_id=config['secrets']['s3_access_key'],
                      aws_secret_access_key=config['secrets']['s3_secret_key'],
                      aws_session_token=config['secrets']['s3_session_token']
                      )

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

if config['metadata']['debug']:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

device = torch.device("cuda:0" if config['config']['use_cuda'] else "cpu")

logging.info("DEVICE: ")
logging.info(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=config['dataset']['local_path'], train=True,
                                        download=config['dataset']['download'], transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(config['config']['batch_size']),
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

S3_MAIN_MODEL_PATH = config['input_model']['s3_key'] + '/' + config['input_model']['model_name']
LOCAL_MAIN_MODEL_PATH = config['input_model']['local_path'] + '/' + config['input_model']['model_name']
download_from_aws(config['input_model']['s3_bucket'], S3_MAIN_MODEL_PATH, LOCAL_MAIN_MODEL_PATH)

net = torch.load(LOCAL_MAIN_MODEL_PATH)
net.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=float(config['config']['lr']), momentum=float(config['config']['momentum']))

for epoch in range(int(config['config']['epochs'])):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

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
MODEL_PATH = config['output']['local_path'] + '/' + MODEL

torch.save(net, MODEL_PATH, _use_new_zipfile_serialization=False)
uploaded = upload_to_aws(MODEL_PATH, config['output']['s3_bucket'], config['output']['s3_key'] + '/' + MODEL)
