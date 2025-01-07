# FaceFusion Ray Serve API

A distributed face swapping service built with Ray Serve and FastAPI.

## Features

- Distributed processing with Ray
- RESTful API with FastAPI
- Asynchronous task processing
- Environment configuration via .env
- Conda environment management
- File cleanup and management
- Detailed logging system

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.11
- Conda package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BillDuke13/facefusion-ray-serve-api.git
cd facefusion-ray-serve-api
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate facefusion-ray-serve-api
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file with your settings
```

## Configuration

Edit `.env` file to configure:

- `UPLOAD_DIR`: Directory for uploaded files
- `OUTPUT_DIR`: Directory for processed outputs
- `FACEFUSION_PATH`: Path to facefusion script
- `SERVICE_HOST`: Service host address
- `SERVICE_PORT`: Service port number
- `RAY_ADDRESS`: Ray cluster address
- `EXECUTION_PROVIDER`: Processing backend (cuda/cpu)

## Usage

1. Start the service:
```bash
python main_serve.py
```

2. API Endpoints:

- POST `/v1/model/facefusion/swap`: Face swap operation
  ```bash
  curl -X POST "http://localhost:9999/v1/model/facefusion/swap" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "source_image=@source.jpg" \
       -F "target_image=@target.jpg"
  ```

- GET `/v1/model/facefusion/status/{task_id}`: Check task status
  ```bash
  curl "http://localhost:9999/v1/model/facefusion/status/{task_id}"
  ```

- GET `/v1/model/facefusion/health`: Service health check
  ```bash
  curl "http://localhost:9999/v1/model/facefusion/health"
  ```

## License
This project is licensed under the Apache 2.0 License. For more details, please refer to the [LICENSE](./LICENSE) file.