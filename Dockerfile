FROM neurodata/graspologic:latest

RUN pip install jupyterlab jupyter
RUN pip install torch==1.12.1 torch_geometric==2.1.0
RUN apt-get update
RUN apt-get install -y g++ gcc
RUN pip install torch_sparse torch_scatter

ARG NB_USER=book
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
RUN  apt-get install -y texlive
RUN pip install git+https://github.com/ebridge2/jupyter-book@master
RUN pip install pyppeteer

RUN apt install -y gconf-service libasound2 libatk1.0-0 \ 
    libatk-bridge2.0-0 libc6 libcairo2 libcups2 libdbus-1-3 \
    libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 \
    libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 \
    libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 \
    libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 \
    libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 \
    lsb-release xdg-utils wget


RUN apt install -y wget vim
RUN apt install -y gnupg gnupg2
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | tee /etc/apt/sources.list.d/google-chrome.list
RUN apt update 
RUN apt install -y google-chrome-stable
RUN pip install seaborn==0.11.2

COPY . ${HOME}
RUN apt install -y gfortran
RUN cd ${HOME} && \
   pip install -r requirements.txt

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
