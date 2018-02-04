FROM tensorflow/tensorflow:latest-py3 

RUN apt-get update
RUN apt-get install python3 python3-dev python3-numpy cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig -y
RUN pip3 install tqdm gym[atari] numpy statsmodels

COPY . /DQN
RUN rm -rf /DQN/.git /DQN/.gitignore /DQN/__pycache__ /DQN/README.md /DQN/.vscode /DQN/Dockerfile /DQN/data/ /DQN/CHTC_start 
RUN chmod +x /DQN/dqn.sh
WORKDIR /DQN

CMD ["/bin/bash"]