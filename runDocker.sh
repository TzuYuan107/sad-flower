#!/bin/bash

xhost +
docker run  -it \
 	    -v /tmp/.X11-unix:/tmp/.X11-unix \
 	    --gpus all \
 	    --runtime nvidia \
 	    -e DISPLAY=$DISPLAY \
 	    --privileged \
		--mount type=bind,source=your-local-path,target=/workspace \
 	    sad_flower \
 	    /bin/bash
