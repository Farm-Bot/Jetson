# Jetson

This repo is the computer vision source code intended to run on jetson nano dev kit.


Clone repo
----------
```
git clone https://github.com/Farm-Bot/Jetson.git
```

Install dependancys
--------------
to install all the dependancys required apart from the defalult pakages avalibe in jetson flavor of ubuntu, run:

```
cd Jetson
pip install -r requirements.txt
```

Run main code
--------------
to initilize the camera and run the CNN algorithm on the video

```
python3 main.py
```
