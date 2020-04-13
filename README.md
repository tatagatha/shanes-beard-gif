# shanes-beard-gif

Fourth week of Covid-19 Quarantine:  
At the start of the Covid-19 Quarantine, we convinced my friend shane to grow a beard.  I wanted to see if I could make a good time laps of the growth.  We don't have a lot to do and I was bored. I needed something to find the pupils and crop the images consistently.  

This is my first project with Open CV and python really. I don't know how to help anyone with any of this. It is clumsily written. Take it as is. It worked this one time with the inputs I had with consistent lighting and image size. And even then it rejected plenty of images that should have worked. [I pulled HEAVILY from this good write up][1].

Set up:
0. Install Python. I was running with python3
0. Install the necessary libraries
```
pip3 install imutils
pip3 install dlib
pip3 install scipy
pip3 install opencv-python
```
0. Install [ffmpeg](https://www.ffmpeg.org/ "I had it installed already, You'll figure it out.")
0. Download the face and eye classifiers.  They are from the Open CV library.
```
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```
0. Run the python script
```
python3 sbg.py
```
0. [Build the gif][2]. You can play with the framerate to your liking.  There lots of other things you can do here if you want to get fancy with ffmpeg.
```
ffmpeg -f image2 -framerate 7 -i ./output/%d.jpg shane.gif
```
0. __Behold__  
![The Start][beard_gif]

[1]:https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
[2]:https://stackoverflow.com/questions/3688870/create-animated-gif-from-a-set-of-jpeg-images
[beard_gif]:./shane.gif
