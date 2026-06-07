# ─────────────────────────────────────────────────────────────
# deploy_cloud.sh  —  Cloud deployment helpers
#
# Supported targets:
#   GCP Cloud Run  (serverless, pay-per-request)
#   AWS EC2        (GPU instance)
#   fly.io         (simple, free tier available)
#
# Usage:
#   bash deploy_cloud.sh gcp   PROJECT_ID
#   bash deploy_cloud.sh aws   INSTANCE_TYPE KEY_NAME
#   bash deploy_cloud.sh fly   APP_NAME
# ─────────────────────────────────────────────────────────────
set -euo pipefail

TARGET="${1:-help}"
shift || true

# ── COMMON ───────────────────────────────────────────────────
IMAGE_NAME="llm-from-scratch"
IMAGE_TAG="latest"

build_image() {
  echo "[docker] Building image ${IMAGE_NAME}:${IMAGE_TAG}..."
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
}

# ════════════════════════════════════════
# GCP Cloud Run
# ════════════════════════════════════════
deploy_gcp() {
  PROJECT_ID="${1:?Usage: $0 gcp PROJECT_ID}"
  REGION="asia-southeast1"          # Bangkok-adjacent
  SERVICE_NAME="llm-api"
  REGISTRY="gcr.io/${PROJECT_ID}/${IMAGE_NAME}"

  echo "🚀  Deploying to GCP Cloud Run (project=${PROJECT_ID})"

  # Authenticate
  gcloud config set project "$PROJECT_ID"

  # Build & push
  build_image
  docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}:${IMAGE_TAG}"
  docker push "${REGISTRY}:${IMAGE_TAG}"

  # Deploy
  gcloud run deploy "$SERVICE_NAME" \
    --image "${REGISTRY}:${IMAGE_TAG}" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --port 8000 \
    --set-env-vars "CHECKPOINT_DIR=/app/runs"

  echo "✅  GCP Cloud Run deployment complete."
  echo "    URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')"
}

# ════════════════════════════════════════
# AWS EC2 (GPU)
# ════════════════════════════════════════
deploy_aws() {
  INSTANCE_TYPE="${1:-g4dn.xlarge}"   # $0.53/hr; 16 GB VRAM
  KEY_NAME="${2:?Usage: $0 aws INSTANCE_TYPE KEY_NAME}"
  AMI_ID="ami-0c55b159cbfafe1f0"     # Deep Learning AMI (Ubuntu 20.04, us-east-1)
  SG_NAME="llm-api-sg"
  REGION="us-east-1"

  echo "🚀  Launching AWS EC2 (${INSTANCE_TYPE})"

  # Create security group (idempotent)
  aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "LLM API" \
    --region "$REGION" 2>/dev/null || true

  SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SG_NAME}" \
    --query "SecurityGroups[0].GroupId" \
    --output text --region "$REGION")

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 \
    --region "$REGION" 2>/dev/null || true

  # User-data script (runs on instance boot)
  cat > /tmp/user_data.sh <<'USERDATA'
#!/bin/bash
set -e
apt-get update && apt-get install -y docker.io git
systemctl start docker
# Clone repo (replace with your actual repo URL)
git clone https://github.com/YOUR_ORG/llm-from-scratch /opt/llm
cd /opt/llm
docker build -t llm-api .
docker run -d -p 8000:8000 --gpus all llm-api
USERDATA

  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data file:///tmp/user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llm-api}]" \
    --query "Instances[0].InstanceId" \
    --output text \
    --region "$REGION")

  echo "  Instance ID: ${INSTANCE_ID}"
  echo "  Waiting for public IP..."
  sleep 15
  PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region "$REGION")

  echo "✅  EC2 launched."
  echo "    SSH:  ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
  echo "    API:  http://${PUBLIC_IP}:8000 (ready after ~3 min)"
}

# ════════════════════════════════════════
# fly.io (simplest option)
# ════════════════════════════════════════
deploy_fly() {
  APP_NAME="${1:?Usage: $0 fly APP_NAME}"

  echo "🚀  Deploying to fly.io (app=${APP_NAME})"

  if ! command -v flyctl &>/dev/null; then
    curl -L https://fly.io/install.sh | sh
    export PATH="$HOME/.fly/bin:$PATH"
  fi

  # Create fly.toml if not present
  if [[ ! -f fly.toml ]]; then
    cat > fly.toml <<FLYTOML
app = "${APP_NAME}"
primary_region = "sin"   # Singapore

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "4gb"
  cpu_kind = "shared"
  cpus = 2
FLYTOML
  fi

  flyctl launch --name "$APP_NAME" --no-deploy --yes 2>/dev/null || true
  flyctl deploy --remote-only

  echo "✅  fly.io deployment complete."
  echo "    URL: https://${APP_NAME}.fly.dev"
}

# ════════════════════════════════════════
# Help
# ════════════════════════════════════════
show_help() {
  cat <<HELP
Usage:
  bash deploy_cloud.sh gcp  <PROJECT_ID>
  bash deploy_cloud.sh aws  <INSTANCE_TYPE> <KEY_NAME>
  bash deploy_cloud.sh fly  <APP_NAME>
  bash deploy_cloud.sh local          # docker compose up

Examples:
  bash deploy_cloud.sh gcp  my-gcp-project-123
  bash deploy_cloud.sh aws  g4dn.xlarge my-keypair
  bash deploy_cloud.sh fly  llm-api-prod
HELP
}

# ════════════════════════════════════════
# Router
# ════════════════════════════════════════
case "$TARGET" in
  gcp)   deploy_gcp "$@" ;;
  aws)   deploy_aws "$@" ;;
  fly)   deploy_fly "$@" ;;
  local) docker compose up --build ;;
  *)     show_help ;;
esac
