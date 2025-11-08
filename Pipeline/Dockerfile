FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Copy everything in your current directory into the image
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Expose GPU info for logging (optional)
ENV CUDA_VISIBLE_DEVICES=0

# Run your pipeline
CMD ["python", "pipeline.py"]


# To RUN the pipeline using docker 
# docker run --gpus all -v "$(pwd)/artifacts:/app/artifacts" anli_pipeline
