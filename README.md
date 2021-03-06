# FL-Client

Client for federated learning demo


## How to run it

```
docker build -t kratemis/fl-server .
```

```
docker run kratemis/fl-server -e CONFIG="{ "operation": "run-fl-task", "dataset": {"download": true,"s3_bucket": "flf-models-us-east-1-artifacts", "s3_key": "source/data", "local_path": "/data/dataset" }, "input_model": {"s3_bucket": "flf-models-us-east-1-artifacts", "s3_key": "main", "local_path": "/data/models", "model_name": "main_model.pt" }, "code": {"type": "git_repo", "url": "https://github.com/SaMnCo/fl-client" }, "config": {"use_cuda": 0,"batch_size": 4,"test_batch_size": 1,"lr": 0.001,"log_interval": 10,"epochs": 5,"momentum": 0.09 }, "output": {"weight_only": true,"s3_bucket": "flf-models-us-east-1-artifacts", "s3_key": "clients", "local_path": "/data/output" }, "metadata": {"debug": true }}"
```
