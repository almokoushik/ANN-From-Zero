FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for output models/plots
RUN mkdir -p models output

# Default command
CMD ["python", "train.py", "--activation", "sigmoid", "--optimizer", "momentum", "--l_rate", "4", "--batch_size", "128"]
