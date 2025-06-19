# PyTorch-KDL: A Kinematics and Dynamics Library for Robotic Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

This project provides a powerful integration of **PyTorch** and the **Orocos Kinematics and Dynamics Library (KDL)**, all conveniently packaged within a Docker container. It is designed for researchers and developers working on robot kinematics, manipulation, and vision-based task training.

**For experimental use only.**

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [Jupyter Lab Environment](#jupyter-lab-environment)
  - [Keyframe Extraction](#keyframe-extraction)
  - [Vision-Based Tasks](#vision-based-tasks)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About The Project

`PyTorch-KDL` bridges the gap between classical robotics and modern deep learning frameworks. By wrapping the robust solvers of KDL within a PyTorch-compatible interface, this library allows for seamless gradient-based optimization of kinematic chains, making it ideal for a new class of robotic learning tasks.

This entire environment is containerized using Docker, ensuring reproducibility and eliminating complex setup procedures.

## Key Features

* **PyTorch-Powered Kinematics:** Leverage PyTorch's automatic differentiation with KDL's reliable kinematics solvers.
* **Dockerized Environment:** Get up and running in minutes with a pre-configured Docker container that includes all necessary dependencies.
* **Jupyter Lab Integration:** Interactively develop and test your robotics algorithms in a web-based Jupyter environment.
* **Vision-Based Manipulation Tasks:** Includes example implementations for vision-based pushing and grasping tasks.
* **Keyframe Extraction:** Tools to extract keyframes from demonstrations for imitation learning.

## Getting Started

Follow these steps to build the Docker image and launch the container.

### Prerequisites

You must have **Docker** installed on your system.

* [Install Docker Engine](https://docs.docker.com/engine/install/)

### Installation and Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/pytorch-kdl.git](https://github.com/your_username/pytorch-kdl.git)
    cd pytorch-kdl
    ```

2.  **Build the Docker image:**
    The `Dockerfile` provided in the root of this project contains all the necessary build steps.
    ```sh
    docker build -t pytorch-kdl .
    ```

3.  **Run the Docker container:**
    This command will start the container, map the necessary ports for Jupyter, and mount the current directory into the container's workspace.
    ```sh
    docker run -it -p 8888:8888 -v "$(pwd)":/workspace pytorch-kdl
    ```

## Usage

### Jupyter Lab Environment

After running the container, you will see a URL in your terminal that looks something like this:
`http://127.0.0.1:8888/lab?token=...`

Copy and paste this URL into your web browser to access the Jupyter Lab interface. All your project files will be available in the `/workspace` directory.

### Keyframe Extraction

The `notebooks/` directory contains examples for using the keyframe extraction tools. These can be used to process robot demonstration data for behavioral cloning or other imitation learning techniques.

### Vision-Based Tasks

Explore the `tasks/` directory to find the implementation of the vision-based pushing and grasping tasks. You can run these examples from the Jupyter notebooks to train and evaluate manipulation policies.

Example:
```python
# Inside a Jupyter notebook
from tasks.pushing import PushingTask

# Initialize and run the pushing task
push_task = PushingTask()
push_task.run_training()
