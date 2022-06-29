# Audio Augmentation

### Supported Augmentation Methods:
* Time Shifting
* Time Stretching
* Pitch Shifting
* Volume Control
* Noise Control

### Supported Augmentation Approaches:
* Sequential Augmentation
* Parallel Augmentation

(_Note: Check out augmentation_demo.ipynb_ and docs/html/index.html for more info.)

### 1) How to Setup the Project?
* Requirements: Docker, Python3.8, dependencies from requirements.txt

#### Using venv
* Install [Python3.8](https://www.python.org/downloads/release/python-3813/)
* Create venv:
  ```
  cd data_augmentation
  python3.8 -m venv aug-venv
  ```
* Activate venv:
  ```
  source aug-venv/bin/activate
  ```
* Install dependencies:
  ```
  python3.8 -m pip install -r requirements.txt
  ```
* Ready to run the project (Section 2)

#### Using Dockerfile
* Install [Docker](https://docs.docker.com/get-docker/)
* To build Docker image use these commands:

    ```
  cd data_augmentation
  docker build -f Dockerfile . -t audio-augmentation
    ```
* Ready to run the project (Section 2)

### 2) How to Run Audio Augmentation?
#### Configuration Settings:
* Set src/config/config.yml `augmentation_job.approach` to `"SEQ"` or `"PAR"` depending on which augmentation approach you want:
* Set `export` to `True` if you want your results (augmented files) to be saved.
* Specify `import_path` and `export_path` for your input and output data folders.
  ```
  augmentation_job:
  approach: "SEQ" #"PAR"
  export: True
  paths:
    import_path: "<Path-To-Your-Local-Folder-With-Input-Data>"
    export_path: "<Path-To-Your-Local-Folder-With-Ourput-Data>"
  ```
#### Run commands
* Run using venv
  ```
  python src/routines/main.py
  ```
* Run using Docker Image
  ```
  docker run -v <Path-To-Your-Local-Folder-With-Input-Data>:/data_augmentation/import_data \
  -v <Path-To-Your-Local-Folder-With-Ourput-Data>:/data_augmentation/export_data \
  -v <Path-To-data_augmentation-Project>/src:/data_augmentation/src \
  -it audio-augmentation
  ```

###  3) How to generate Sphinx documentation?
* To clean and create `index.html` page run these commands:
  ```
  cd docs
  make clean
  make html
  ```