# Cuda Runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

#Non interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# working directory
WORKDIR /app

# dependencines
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0
# copy source and model files
COPY . .


#entrypoint
ENTRYPOINT ["python3", "inference.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]