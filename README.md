# facefusion-ray-serve-api

## Overview
This repository provides a Ray Serve and FastAPI-based service for face swapping. It uses the FaceFusion tool to process requests asynchronously, manage file storage, and track task statuses.

## Features
- Distributed face fusion tasks via Ray Serve.
- Asynchronous processing with automatic file management.
- Task status queries, health checks, and logs.

## Installation
1. Clone this repository.
2. Install requirements:  
   pip install -r requirements.txt
3. Ensure Ray is initialized with the correct address in config.py.

## Usage
1. Start the service:  
   python main_serve.py
2. Service runs at:  
   http://0.0.0.0:9999/v1/model/facefusion

## API Endpoints
• POST /swap  
  Accepts two image files (source and target) and returns a FaceFusionResponse.  
• GET /status/{task_id}  
  Returns the current status and result path for a given task.  
• GET /health  
  Reports service health.  
• GET /stats  
  Provides basic usage statistics.

## Contributing
1. Fork the repo and make a feature branch.
2. Submit a pull request.

## License
This project is licensed under the Apache 2.0 License. For more details, please refer to the [LICENSE](./LICENSE) file.