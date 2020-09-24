xhost local:root
docker run -ti --rm -e DISPLAY=unix"$DISPLAY" \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v /home/mvu/Downloads/video:/video \
      --runtime=nvidia 3stages \
      ./run_script.sh -gpu 0 ./demo_cnn_lstm.py --video-file=/video/LeftVideoSN001_comp.avi