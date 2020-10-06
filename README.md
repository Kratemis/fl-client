# fl-client

parser.add_argument('--download-folder', required=True)
parser.add_argument('--local-folder', required=True)
parser.add_argument('--config-file', required=True)
parser.add_argument('--job-id', required=True)
parser.add_argument('--bucket', required=True)
parser.add_argument('--s3-access-key', required=True)
parser.add_argument('--s3-secret-key', required=True)
parser.add_argument('--main-model', required=True)
parser.add_argument('-d', '--debug', help="Debug mode for the script")


## Arguments
* Download Folder (Path where CIFAR dataset is downloaded)
* Local Folder (Path where local model is generated)
* Config File (Configuration file path for the app)
* Job Id (Unique ID for the process)
* Bucket (S3 Bucket name)
* S3 Access Key
* S3 Secret Key
* Main Model Path (Path for the main model)
* Debug (Debug mode)
