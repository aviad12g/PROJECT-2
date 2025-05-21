FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gsutil
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the remaining application files
COPY . .

# Make sure the results directory exists
RUN mkdir -p results/figures

# Set default GCS bucket (can be overridden at build time)
ARG GCS_BUCKET="arc-easy-bucket"
ENV GCS_BUCKET=$GCS_BUCKET

# Run the evaluation pipeline
ENTRYPOINT ["bash","-lc","\
    python prepare_arc.py && \
    python run_dcbs_eval.py --config configs/study_config.yaml --inject_reasoning && \
    python plot_results.py --input_csv results/exp_sweeps.csv && \
    gsutil cp results/arc_easy_dev.jsonl gs://${GCS_BUCKET}/arc_easy_dev.jsonl && \
    gsutil cp results/exp_sweeps.csv gs://${GCS_BUCKET}/dcbs-results/exp_sweeps.csv && \
    gsutil cp -r results/figures gs://${GCS_BUCKET}/dcbs-results/figures/ \
"] 