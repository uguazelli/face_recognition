# Face Recognition

## Overview

This repository contains two applications that identify and recognize faces:

1. `id_single.py`: Given an image, this script will display "matched" if a face is recognized, or "no match" if it doesn't.
2. `id_multiple.py`: Given a directory with images, this script will display the names of the persons that match the faces in the images.

## Installation

### Setting Up a Virtual Environment

It is recommended to use a virtual environment for this project. You can create one using the following command:

```bash
python -m venv <path_to_new_virtual_environment>
```

To activate the virtual environment:

- On Windows:

```bash
<venv>\Scripts\activate.bat
```

- On Linux:

```bash
source <venv>/bin/activate
```

### Installing Dependencies

This project uses the OpenCV and face_recognition libraries. The face_recognition library requires dlib, which can be challenging to install. To install dlib, find the corresponding file in the `dlib_windows` folder that matches your installed Python version and run the following command (replace `<python_version>` with your Python version):

```bash
pip install dlib-19.24.1-cp3<python_version>-cp3<python_version>-win_amd64
```

Then, install all the required libraries:

```bash
pip install -r requirements.txt
```

_Note: The dlib installation process resolves the following issues:_

- Failed to build dlib
- Failed building wheel for dlib
- face recognition installation error
- pip install dlib error
- pip install cmake error
- pip install face-recognition error
- dlib library installation problem
- face recognition python library installation problem
- How to install dlib
- How to install face recognition library
- How to install cmake library
- Error: legacy install failure
- Not supported wheel

## Execution

Running the scripts is straightforward:

1. Add the images to the `reference_photos` directory.
2. Name the images according to the desired output.
3. Run the script with `python <file_name>`.

For a detailed walkthrough, you can watch this [video tutorial](https://youtu.be/qImLubzBOWg).

```

```
