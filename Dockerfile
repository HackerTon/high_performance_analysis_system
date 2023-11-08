FROM continuumio/miniconda3:23.9.0-0

ARG MODE

RUN echo MODE: $MODE

WORKDIR /app

RUN apt update
RUN apt install -y ffmpeg libgl1-mesa-glx
RUN conda install libmamba
RUN conda config --set solver libmamba

COPY requirements.txt .
COPY environment_cpu.yml .
COPY environment_gpu.yml .

RUN if [ "$MODE" = "CPU" ]; then \
    conda env create -f environment_cpu.yml; \
    elif [ "$MODE" = "GPU" ]; then \
    conda env create -f environment_cpu.yml; \
    else \
    echo "No valid model provided MODE: (GPU | GPU)"; \
    fi

SHELL ["conda", "run", "-n", "pytorch", "/bin/bash", "-c"]
RUN python --version

COPY src ./src
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch", "uvicorn", "--app-dir=src/", "app:app", "--host", "0.0.0.0"]