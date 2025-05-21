#!/bin/bash
set -e

# Default values
GCS_BUCKET="arc-easy-bucket"
REGION="us-central1"
REPO="dcbs-images"
IMAGE_NAME="dcbs-eval"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket=*)
      GCS_BUCKET="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure you're in your project dir
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
  echo "Please set your Google Cloud project:"
  echo "  gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

# Enable required services
echo "Enabling Artifact Registry and Vertex AI services..."
gcloud services enable artifactregistry.googleapis.com \
  aiplatform.googleapis.com --quiet

# Configure Docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Create Artifact Registry repo if missing
echo "Creating Docker repo (if needed)..."
gcloud artifacts repositories create ${REPO} \
  --repository-format=docker \
  --location=${REGION} \
  --description="DCBS evaluation images" \
  --quiet || true

# Ensure GCS bucket exists
echo "Checking GCS bucket: $GCS_BUCKET"
if ! gsutil ls -b gs://$GCS_BUCKET >/dev/null 2>&1; then
  echo "  Bucket not found, creating..."
  gsutil mb -l ${REGION} gs://$GCS_BUCKET
else
  echo "  Bucket already exists"
fi

# Build & push Docker image
IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:latest
echo "Building and pushing image ${IMAGE_URI}..."
docker build \
  --build-arg GCS_BUCKET=$GCS_BUCKET \
  -t $IMAGE_URI .
docker push $IMAGE_URI

# Submit a CPU‐only Custom Job
echo "Submitting Vertex AI Custom Job (CPU only)..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=dcbs-full-eval \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=$IMAGE_URI \
  --quiet

echo "✅ Job submitted. Monitor with:"
echo "    gcloud ai custom-jobs list --region=${REGION}"
echo
echo "Once complete, fetch results with:"
echo "    gsutil ls gs://$GCS_BUCKET/dcbs-results/"
echo "    gsutil cp -r gs://$GCS_BUCKET/dcbs-results ."
