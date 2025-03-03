FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /code

RUN apt update && \
    apt install -y git libegl1 libegl-dev && \
    apt clean

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . .

ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl

EXPOSE 7860

ENTRYPOINT ["python", "app.py"]