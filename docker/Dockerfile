FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
