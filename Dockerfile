FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y python3-pip libjpeg-dev zlib1g-dev xvfb ffmpeg graphviz python3-opengl && apt-get clean

# Downgrade pip to avoid the gym installation issue
RUN pip install setuptools==65.5.0 wheel==0.38.4
RUN pip install tensorflow-quantum==0.7.3 gym==0.18.0 jupyterlab pyvirtualdisplay pydot

# Expose port and create entry point for jupyterlab
EXPOSE 8888
CMD ["jupyter-lab", "--notebook-dir='/workspace'", "--NotebookApp.token='qrl'", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
