# Optimal Inventory Management System - Deployment Guide

This document provides instructions for deploying the Optimal Inventory Management System in a production environment.

## System Requirements

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL database
- Minimum 4 CPU cores, 8GB RAM for standard deployments
- 20GB disk space

## Architecture Components

The system consists of several containerized services:

1. **Core Optimization Engine** - Dynamic programming solver
2. **Web Dashboard** - User interface for visualization and configuration
3. **Forecasting Service** - Time series analysis and demand modeling
4. **API Gateway** - Central access point for all services
5. **Data Connectors** - Interfaces with ERP, POS, and other systems

## Deployment Options

### 1. Local Development Deployment

```bash
# Clone the repository
git clone https://github.com/your-org/optimal-inventory-management.git
cd optimal-inventory-management

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the development environment
docker-compose -f docker-compose.dev.yml up
```

### 2. Production Deployment with Docker Compose

```bash
# Clone the repository
git clone https://github.com/your-org/optimal-inventory-management.git
cd optimal-inventory-management

# Set up environment variables
cp .env.example .env.prod
# Edit .env.prod with your production configuration

# Start the production environment
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Kubernetes Deployment

Kubernetes manifests are provided in the `kubernetes/` directory.

```bash
# Ensure kubectl is configured for your cluster

# Create namespace
kubectl create namespace inventory-optimization

# Apply ConfigMaps and Secrets
kubectl apply -f kubernetes/configmaps.yaml
kubectl apply -f kubernetes/secrets.yaml

# Deploy services
kubectl apply -f kubernetes/services/

# Deploy ingress
kubectl apply -f kubernetes/ingress.yaml

# Verify deployments
kubectl get pods -n inventory-optimization
```

## Configuration

### Environment Variables

Key environment variables that need to be configured:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/inventory` |
| `SECRET_KEY` | Application secret key | `super-secret-random-string` |
| `ERP_API_URL` | URL of the ERP API | `https://erp.company.com/api/v1` |
| `ERP_API_KEY` | API key for ERP access | `api-key-123456` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WORKERS` | Number of optimization workers | `4` |

### Database Setup

```bash
# Run database migrations
docker-compose exec api flask db upgrade

# Load initial data (optional)
docker-compose exec api flask seed-data
```

## Integration with External Systems

### ERP System Integration

1. **API Configuration**: Set up the ERP API credentials in the `.env` file.
2. **Data Mapping**: Configure the data mapping in `config/erp_mapping.json`.
3. **Testing**: Test the connection with:
   ```bash
   docker-compose exec api flask test-erp-connection
   ```

### POS System Integration

1. **API Configuration**: Set up the POS API credentials in the `.env` file.
2. **Configure Webhooks**: Set up webhooks for real-time sales data in your POS system.
3. **Testing**: Test the connection with:
   ```bash
   docker-compose exec api flask test-pos-connection
   ```

## Scaling Considerations

### Horizontal Scaling

The system is designed to scale horizontally:

- **API Gateway**: Add more replicas behind a load balancer
- **Optimization Engine**: Add more worker nodes
- **Database**: Set up read replicas for heavy reporting workloads

### Vertical Scaling

For larger optimization problems:

- Increase RAM allocation for optimization nodes
- Increase CPU allocation for forecasting nodes

## Monitoring and Maintenance

### Health Checks

Endpoints available for health monitoring:

- `/health` - Basic health check
- `/health/optimization` - Optimization engine status
- `/health/database` - Database connection status
- `/health/erp` - ERP connection status

### Logs

All services log to stdout/stderr, which is captured by Docker. In production:

1. Configure a log aggregation system (ELK, Datadog, etc.)
2. Set up alerts for error conditions

### Backups

1. **Database Backups**: 
   ```bash
   docker-compose exec db pg_dump -U username dbname > backup.sql
   ```

2. **Configuration Backups**:
   ```bash
   docker-compose exec api flask export-config > config-backup.json
   ```

## Troubleshooting

### Common Issues

1. **Optimization Engine Timeouts**:
   - Increase timeout settings in `config/optimization.json`
   - Reduce problem size or increase resources

2. **ERP Connection Failures**:
   - Check network connectivity
   - Verify API credentials
   - Check ERP system is available

3. **High Memory Usage**:
   - Reduce the problem state space
   - Increase container memory limits
   - Implement pagination for large result sets

## Security Considerations

1. **API Security**:
   - Use HTTPS for all connections
   - Implement API key or OAuth2 authentication
   - Set up rate limiting

2. **Data Security**:
   - Encrypt sensitive data at rest
   - Implement role-based access control
   - Regularly audit access logs

3. **Network Security**:
   - Use a private network for inter-service communication
   - Implement firewalls to restrict access
   - Use VPNs for remote administration

## Upgrading

1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Update containers:
   ```bash
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. Run migrations if needed:
   ```bash
   docker-compose exec api flask db upgrade
   ```

## Support and Resources

- Documentation: [https://docs.inventory-optimization.company.com](https://docs.inventory-optimization.company.com)
- Support: [support@company.com](mailto:support@company.com)
- Issue Tracker: [https://github.com/your-org/optimal-inventory-management/issues](https://github.com/your-org/optimal-inventory-management/issues) 