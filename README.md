# FL-Client

Client for federated learning demo

## Arguments
* --s3-client-models-folder (S3 folder for client model)
* --s3-main-models-folder (S3 folder for main model)
* --local-folder (Local folder)
* --initial-main-model (Initial main model)
* --config-file (Configuration file with ML parameter)
* --job-id (Unique Job Id)
* --clients-bucket (Bucket name for client model)
* --main-bucket (Bucket name for main model)
* --s3-access-key (Credentials for AWS)
* --s3-secret-key (Credentials for AWS)
* --debug (Debug mode)

## Example
```bash
python main.py --config-file '{"use_cuda": 0, "batch_size": 3, "test_batch_size": 1, "lr": 0.001, "log_interval": 10, "epochs": 10, "momentum": 0.09}' --s3-client-models-folder "clients" --s3-main-models-folder "main" --initial-main-model "main_model.pt" --local-dataset-folder "./dataset" --local-client-models-folder "./storage" --local-main-model-folder "./storage" --job-id 4245245 --clients-bucket "MY_BUCKET_NAME" --main-bucket "ANOTHER_BUCKET_NAME"
```
