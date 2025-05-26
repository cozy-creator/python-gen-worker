# Start from the specific pre-built base worker image
FROM cozycreator/base-worker:cuda126-py312-base-v1

# Set the working directory
WORKDIR /workspace

# Copy the specific requirements file for this function set
COPY requirements.txt .

# Install ONLY the additional dependencies using uv (inherited from base)
# Provide the private index URL again in case these deps need base packages or specific versions from there.
# If not needed, you might be able to remove the --index-url here.
RUN uv pip install \
    --system \
    --index-url http://137.184.153.104:3141/root/mydev/+simple \
    -r requirements.txt

# Copy the Python file containing the decorated function(s)
COPY src/image_gen ./image_gen

# Set the environment variable telling the worker which Python module
# contains the decorated functions.
ENV USER_MODULES=image_gen
ENV ENABLE_DEFAULT_MODEL_MANAGER=true