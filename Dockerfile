# FROM --platform=amd64 continuumio/miniconda3:23.9.0-0
FROM continuumio/miniconda3:23.9.0-0

WORKDIR /app

RUN apt update
RUN apt install -y ffmpeg libgl1-mesa-glx

COPY environment.yml .
COPY requirements.txt .
RUN conda install libmamba
RUN conda config --set solver libmamba
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]
RUN python --version

COPY src ./src
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch", "python", "src/main.py"]