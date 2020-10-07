# FL-Client

Client for federated learning demo

## Arguments
* Download Folder (Path where CIFAR dataset is downloaded)
* Local Folder (Path where local model is generated)
* Config File (Configuration file path for the app)
* Job Id (Unique ID for the process)
* Bucket (S3 Bucket name)
* S3 Access Key
* S3 Secret Key
* S3 Folder
* Main Model Path (Path for the main model)
* Debug (Debug mode)

## Example
```bash
python main.py --config-file '{"use_cuda": 0, "batch_size": 3, "test_batch_size": 1, "lr": 0.001, "log_interval": 10, "epochs": 10, "momentum": 0.09}' --s3-client-models-folder "clients" --s3-main-models-folder "main" --local-dataset-folder "./dataset" --local-client-models-folder "./storage" --local-main-model-folder "./storage" --job-id 4245245 --clients-bucket "MY_BUCKET_NAME" --main-bucket "ANOTHER_BUCKET_NAME"
```
