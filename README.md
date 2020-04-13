# shanes-beard-gif

Fourth week of Covid-19 Quarantine:  
At the start of the Covid-19 Quarantine, we convinced my friend shane to grow a beard.  I wanted to see if I could make a good time laps of the growth.  We don't have a lot to do and I was bored. I needed something to find the pupils and crop the images consistently.  

This is my first project with Open CV and python really. I don't know how to help anyone with any of this. It is clumsily written. Take it as is. It worked this one time with the inputs I had with consistent lighting and image size. And even then it rejected plenty of images that should have worked. [I pulled HEAVILY from this good write up][1].

###Set up:  
1. Install Python. I was running with python3
1. Install the necessary libraries
    ```
    pip3 install imutils
    pip3 install dlib
    pip3 install scipy
    pip3 install opencv-python
    ```
1. Install [ffmpeg](https://www.ffmpeg.org/ "I had it installed already, You'll figure it out.")
1. Download the face and eye classifiers.  They are from the Open CV library.  
    ```
    wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
    wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
    ```
1. Run the python script  
    ```
    python3 sbg.py
    ```  
 1. [Build the gif][2]. Then [Make a better gif][bettergif].  You can play with the framerate to your liking.  There lots of other things you can do here if you want to get fancy with ffmpeg.
    ```
    ffmpeg -f image2 -framerate 7 -i ./output/%d.jpg shane.gif

    # looks fine but after posting it I was ashamed of all the extensive dithering.
    # Let's do better.

    # make a video first
    ffmpeg -framerate 7 -i ./output/%d.jpg shane.avi

    # make a palette. Its unclear what those scale and fps do when making a pallete.
    ffmpeg -i shane.avi -vf fps=7,scale=300:-1:flags=lanczos,palettegen palette.png

    # Then use that palette as an input. I do know that the scale is the output width of the gif. The rest is unclear.
    ffmpeg -i shane.avi -i palette.png  -filter_complex "fps=6,scale=300:-1:flags=lanczos[x];[x][1:v]paletteuse" better_shane.gif
    ```
1. __Behold__  
![The Start][beard_gif]
![The Finish][better_beard_gif]


[1]:https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
[2]:https://stackoverflow.com/questions/3688870/create-animated-gif-from-a-set-of-jpeg-images
[beard_gif]:./shane.gif
[better_beard_gif]:better_shane.gif
[bettergif]:https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483
