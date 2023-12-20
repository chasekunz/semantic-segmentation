
# Semantic Segmentation Project

## Overview
This project demonstrates the training of a DeepLabV3 model on the KITTI dataset for semantic segmentation in automotive environments. The aim is to segment distinct areas efficiently and accurately.

## Getting Started

### Prerequisites
- Docker
- Download required datasets

### Downloading the KITTI Dataset
To download the KITTI Semantic Segmentation Pixel-Level Benchmark dataset, visit the [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_semantics.php) and follow their instructions for downloading.

### Setup
1. **Clone the Repository**: Clone this GitHub repository to get started.
   ```bash
   git clone git@github.com:chasekunz/semantic-segmentation.git
   cd semantic-segmentation
   ```

2. **Download Data**: After downloading the KITTI dataset, unzip it and place the `data_semantics` folder in the `data` directory of this repository.

3. **Configure Environment**: Use the provided `environment.yaml` to set up your Python environment. This can be done using Conda:
   ```bash
   conda env create -f environment.yaml
   ```


### Docker Configuration
1. **Build Docker Image**: To containerize the application, build the Docker image.
   ```bash
   docker compose build
   ```
2. **Run Docker Container**: Start the Docker container. This will also start a Jupyter server, which can be accessed through the links output by the container.
   ```bash
   docker compose up
   ```
3. **Shutdown**: To quit and remove the container, use:
   ```bash
   docker compose down
   ```

## Usage
Refer to the `semantic-segmentation.ipynb` for detailed instructions on how to train and evaluate the model.

## Contributing
Contributions are welcome! Please read our contributing guidelines for details.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
