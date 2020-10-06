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

parser.add_argument('--download-folder', required=True)
parser.add_argument('--local-folder', required=True)
parser.add_argument('--s3-folder', required=True)
parser.add_argument('--config-file', required=True)
parser.add_argument('--job-id', required=True)
parser.add_argument('--bucket', required=True)
parser.add_argument('--s3-access-key', required=True)
parser.add_argument('--s3-secret-key', required=True)
parser.add_argument('--main-model-path', required=True)
parser.add_argument('-d', '--debug', help="Debug mode for the script")
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

MODEL = int(time.time()) + '_model.pt'
MODEL_PATH = args.local_folder + '/' + MODEL

MAIN_MODEL_PATH = args.main_model_path


def load_config():
    logging.info('Loading config')
    with open(args.config_file) as config_file:
        data = json.load(config_file)
        logging.info('Config loaded')
    return data


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


config = load_config()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=args.download_folder, train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(config['batch_size']),
                                          shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = torch.load(MAIN_MODEL_PATH)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=float(config['lr']), momentum=float(config['momentum']))

for epoch in range(int(config['epochs'])):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
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
torch.save(net, MODEL_PATH, _use_new_zipfile_serialization=False)
uploaded = upload_to_aws(MODEL_PATH, args.bucket, args.s3_folder + '/' + MODEL)
