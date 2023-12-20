# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Set PYTHONPATH environment variable to include /usr/src/app
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

# Expose port for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
