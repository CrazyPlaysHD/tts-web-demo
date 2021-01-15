FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN echo "Acquire { HTTP::proxy \"http://192.168.193.12:3128\"; HTTPS::proxy \"http://192.168.193.12:3128\";}" > /etc/apt/apt.conf.d/proxy.conf
RUN touch /etc/apt/apt.conf.d/99verify-peer.conf \
&& echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update -y
RUN apt-get install -y libsndfile-dev
# RUN pip install inflect==4.1.0 Unidecode pillow pronouncing==0.2.0 \
# pandas==1.0.3 num2words tqdm==4.46.0 unidecode ipython numba==0.48  tb-nightly future \
#  --proxy=192.168.193.12:3128

# ADD libs /libs/
# WORKDIR /libs/
# RUN pip install torch-1.6.0+cu101-cp36-cp36m-linux_x86_64.whl
# RUN pip install torchvision-0.7.0+cu101-cp36-cp36m-linux_x86_64.whl

COPY . /opt/data
WORKDIR /opt/data
RUN pip install libs/torch-1.6.0+cu101-cp36-cp36m-linux_x86_64.whl
RUN pip install libs/torchvision-0.7.0+cu101-cp36-cp36m-linux_x86_64.whl
RUN pip install -r requirements.txt --proxy=192.168.193.12:3128

# RUN pip install tensorflow-gpu==1.13.1 tensorflow-estimator==1.13.0 tensorboard==2.0 tensorboard-plugin-wit==1.7.0 numpy==1.18.1 tornado librosa==0.7.2 \
#  librosa==0.7.2 scipy==1.4.1 Unidecode==1.1.2  inflect==4.1.0 Flask==1.1.2  Flask-JSON==0.3.4 pydub==0.24.1 requests==2.25.1 --proxy=192.168.193.12:3128


