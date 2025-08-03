# Deployment Guide

This guide covers deployment options for the FHFA Seasonal Adjustment Pipeline.

## Prerequisites

- Docker 20.10+
- Docker Compose 1.29+
- 16GB RAM minimum
- 10GB disk space

## Deployment Options

### 1. Local Development

For local development and testing:

```bash
# Start all services
docker-compose up

# Run pipeline only
docker-compose run fhfa-pipeline python main.py --mode full

# Access Jupyter notebook
docker-compose up notebook
# Navigate to http://localhost:8888
```

### 2. Single Server Deployment

For deployment on a single server:

```bash
# Build production image
docker build -t fhfa-seasonal-adjustment:prod -f Dockerfile .

# Run with production settings
docker run -d \
  --name fhfa-pipeline \
  -e FHFA_ENV=production \
  -e FHFA_LOG_LEVEL=WARNING \
  -v /data/fhfa/input:/app/data \
  -v /data/fhfa/output:/app/output \
  -v /data/fhfa/cache:/app/.cache \
  fhfa-seasonal-adjustment:prod \
  python main.py --mode full --report
```

### 3. Container Orchestration

#### Kubernetes Deployment

Create deployment manifest:

```yaml
# fhfa-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fhfa-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fhfa-pipeline
  template:
    metadata:
      labels:
        app: fhfa-pipeline
    spec:
      containers:
      - name: pipeline
        image: fhfa-seasonal-adjustment:prod
        env:
        - name: FHFA_ENV
          value: "production"
        - name: FHFA_PARALLEL_ENABLED
          value: "true"
        - name: FHFA_N_JOBS
          value: "8"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: output
          mountPath: /app/output
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: fhfa-data-pvc
      - name: output
        persistentVolumeClaim:
          claimName: fhfa-output-pvc
```

Deploy:
```bash
kubectl apply -f fhfa-deployment.yaml
```

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml fhfa-stack
```

### 4. Cloud Deployment

#### AWS ECS

1. Push image to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker tag fhfa-seasonal-adjustment:prod $ECR_URI/fhfa-seasonal-adjustment:prod
docker push $ECR_URI/fhfa-seasonal-adjustment:prod
```

2. Create task definition with appropriate resource allocations
3. Deploy as ECS service or Fargate task

#### Azure Container Instances

```bash
# Create resource group
az group create --name fhfa-rg --location eastus

# Create container instance
az container create \
  --resource-group fhfa-rg \
  --name fhfa-pipeline \
  --image fhfa-seasonal-adjustment:prod \
  --cpu 4 \
  --memory 16 \
  --environment-variables FHFA_ENV=production
```

#### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/$PROJECT_ID/fhfa-seasonal-adjustment

# Deploy to Cloud Run
gcloud run deploy fhfa-pipeline \
  --image gcr.io/$PROJECT_ID/fhfa-seasonal-adjustment \
  --platform managed \
  --memory 16Gi \
  --cpu 8 \
  --timeout 3600
```

## Production Configuration

### Environment Variables

Essential production settings:

```bash
# Performance
FHFA_PARALLEL_ENABLED=true
FHFA_N_JOBS=8
FHFA_CHUNK_SIZE=5000
FHFA_MEMORY_LIMIT_GB=32.0

# Logging
FHFA_LOG_LEVEL=WARNING
FHFA_LOG_FILE=/var/log/fhfa/pipeline.log

# Monitoring
FHFA_ENABLE_MONITORING=true
FHFA_METRICS_EXPORT_PATH=/metrics/pipeline_metrics.json

# Data Quality
FHFA_MIN_OBSERVATIONS=20
FHFA_MAX_MISSING_RATE=0.1
FHFA_VALIDATION_TOLERANCE=0.0001
```

### Resource Requirements

| Workload | CPU | Memory | Storage |
|----------|-----|--------|---------|
| National | 2 cores | 4GB | 1GB |
| State-level | 4 cores | 8GB | 5GB |
| Top 100 MSAs | 8 cores | 16GB | 10GB |
| All MSAs | 16 cores | 32GB | 50GB |

### Health Checks

Configure health endpoints:

```python
# health_check.py
from src.utils import health_checker

# Register checks
health_checker.register_check("database", check_database_connection)
health_checker.register_check("filesystem", check_filesystem_access)
health_checker.register_check("memory", check_memory_usage)

# Run checks
health_status = health_checker.run_checks()
```

### Monitoring Setup

1. **Prometheus Integration**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'fhfa-pipeline'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

2. **Grafana Dashboard**
- Import dashboard from `monitoring/grafana/dashboard.json`
- Configure alerts for:
  - Pipeline failures
  - High memory usage (>80%)
  - Long processing times
  - Data validation errors

### Backup and Recovery

1. **Data Backup**
```bash
# Backup script
#!/bin/bash
BACKUP_DIR=/backups/fhfa/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup input data
rsync -av /data/fhfa/input/ $BACKUP_DIR/input/

# Backup results
rsync -av /data/fhfa/output/ $BACKUP_DIR/output/

# Backup metrics
rsync -av /data/fhfa/.metrics/ $BACKUP_DIR/metrics/
```

2. **Recovery Procedure**
- Restore data from backup location
- Verify data integrity
- Resume processing from last checkpoint

### Security Considerations

1. **Container Security**
- Run as non-root user
- Use minimal base image
- Scan for vulnerabilities regularly

2. **Data Security**
- Encrypt sensitive data at rest
- Use secure connections for data transfer
- Implement access controls

3. **Network Security**
- Use private networks for internal communication
- Implement firewall rules
- Enable TLS for external endpoints

## Troubleshooting

### Common Issues

1. **Out of Memory**
- Increase container memory limits
- Reduce `FHFA_CHUNK_SIZE`
- Enable disk-based caching

2. **Slow Performance**
- Check CPU allocation
- Verify parallel processing is enabled
- Monitor disk I/O

3. **Pipeline Failures**
- Check logs: `docker logs fhfa-pipeline`
- Verify data format and quality
- Review error metrics

### Debugging

Enable debug mode:
```bash
docker run -e FHFA_LOG_LEVEL=DEBUG fhfa-seasonal-adjustment:prod python main.py
```

Access container shell:
```bash
docker exec -it fhfa-pipeline /bin/bash
```

## Maintenance

### Regular Tasks

1. **Log Rotation**
- Logs rotate automatically at 10MB
- Retain logs for 7 days
- Archive older logs to cold storage

2. **Cache Cleanup**
```bash
# Clean cache older than 30 days
find /app/.cache -mtime +30 -delete
```

3. **Metrics Export**
```bash
# Export and archive metrics monthly
python export_metrics.py --format json --output /archive/metrics_$(date +%Y%m).json
```

### Updates and Upgrades

1. **Update Dependencies**
```bash
# Update requirements
pip-compile requirements.in -U

# Rebuild image
docker build -t fhfa-seasonal-adjustment:new .

# Test new image
docker run fhfa-seasonal-adjustment:new pytest

# Deploy if tests pass
docker tag fhfa-seasonal-adjustment:new fhfa-seasonal-adjustment:prod
```

2. **Rolling Updates**
- Deploy new version alongside old
- Gradually shift traffic
- Monitor for issues
- Rollback if necessary

## Support

For deployment issues:
- Check documentation at `/docs`
- Review logs and metrics
- Contact support team