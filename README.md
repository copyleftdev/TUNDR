<div align="center">
  <img src="media/logo.png" alt="TUNDR MCP Logo" width="200">
  <h1>TUNDR MCP Optimization Server</h1>
  
  [![Go Report Card](https://goreportcard.com/badge/github.com/copyleftdev/mcp-optimization)](https://goreportcard.com/report/github.com/copyleftdev/mcp-optimization)
  [![Go Reference](https://pkg.go.dev/badge/github.com/copyleftdev/mcp-optimization.svg)](https://pkg.go.dev/github.com/copyleftdev/mcp-optimization)
  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  [![Tests](https://github.com/copyleftdev/mcp-optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/copyleftdev/mcp-optimization/actions)
  [![Coverage Status](https://coveralls.io/repos/github/copyleftdev/mcp-optimization/badge.svg?branch=main)](https://coveralls.io/github/copyleftdev/mcp-optimization?branch=main)

  A high-performance optimization server implementing the Model Context Protocol (MCP) for mathematical optimization tasks, with a focus on Bayesian Optimization using Gaussian Processes. Part of the CopyleftDev ecosystem.
</div>

## 🌟 Features

### 🎯 Bayesian Optimization
- Multiple kernel support (Matern 5/2, RBF, Custom)
- Expected Improvement acquisition function (with support for Probability of Improvement and UCB)
- Support for both minimization and maximization problems
- Parallel evaluation of multiple points
- Constrained optimization support
- MCP-compliant API endpoints

### 🛠️ Robust Implementation
- Comprehensive test coverage
- Graceful error handling and recovery
- Detailed structured logging with [zap](https://github.com/uber-go/zap)
- Context-aware cancellation and timeouts
- Memory-efficient matrix operations
- MCP protocol compliance

### 🚀 Performance Optimizations
- Fast matrix operations with [gonum](https://gonum.org/)
- Efficient memory management with object pooling
- Optimized Cholesky decomposition with fallback to SVD
- Parallel batch predictions

### 📊 Monitoring & Observability
- Prometheus metrics endpoint
- Structured logging in JSON format
- Distributed tracing support (OpenTelemetry)
- Health check endpoints
- Performance profiling endpoints

## Features

- **Bayesian Optimization** with Gaussian Processes
  - Multiple kernel support (Matern 5/2, RBF)
  - Expected Improvement acquisition function
  - Numerical stability with Cholesky decomposition and SVD fallback
  - Support for both minimization and maximization problems
  - Parallel evaluation of multiple points

- **Robust Implementation**
  - Comprehensive test coverage (>85%)
  - Graceful error handling and recovery
  - Detailed logging with structured logging (zap)
  - Context-aware cancellation

- **API & Integration**
  - JSON-RPC 2.0 over HTTP/2 interface
  - RESTful endpoints for common operations
  - OpenAPI 3.0 documentation
  - gRPC support (planned)

- **Monitoring & Observability**
  - Prometheus metrics endpoint
  - Structured logging
  - Distributed tracing (OpenTelemetry)
  - Health checks

- **Scalability**
  - Stateless design
  - Horizontal scaling support
  - Multiple storage backends (SQLite, PostgreSQL)
  - Caching layer (Redis)

## 🚀 Quick Start

### MCP Protocol Support

This server implements the Model Context Protocol (MCP) for optimization tasks. The MCP provides a standardized way to:

- Define optimization problems
- Submit optimization tasks
- Monitor optimization progress
- Retrieve optimization results

The server exposes MCP-compatible endpoints for seamless integration with other MCP-compliant tools and services.

### Prerequisites

- Go 1.21 or later
- Git (for version control)
- Make (for development tasks)
- (Optional) Docker and Docker Compose for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/copyleftdev/mcp-optimization.git
cd mcp-optimization

# Install dependencies
go mod download

# Build the server
go build -o bin/server ./cmd/server
```

### Running the Server

```bash
# Start the server with default configuration
./bin/server

# Or with custom configuration
CONFIG_FILE=config/local.yaml ./bin/server
```

### Using Docker

```bash
# Build the Docker image
docker build -t tundr/mcp-optimization-server .

# Run the container
docker run -p 8080:8080 tundr/mcp-optimization-server
```

## 📚 Documentation

### MCP Integration

The server implements the following MCP endpoints:

- `POST /mcp/optimize` - Submit a new optimization task
- `GET /mcp/status/:id` - Check the status of an optimization task
- `GET /mcp/result/:id` - Get the results of a completed optimization
- `DELETE /mcp/task/:id` - Cancel a running optimization task

### API Reference

Check out the [API Documentation](https://pkg.go.dev/github.com/copyleftdev/mcp-optimization) for detailed information about the available methods and types.

### Example: Basic Usage

```go
package main

import (
	"context"
	"fmt"
	"math"
	
	"github.com/tundr/mcp-optimization-server/internal/optimization"
	"github.com/tundr/mcp-optimization-server/internal/optimization/bayesian"
	"github.com/tundr/mcp-optimization-server/internal/optimization/kernels"
)

func main() {
	// Define the objective function (to be minimized)
	objective := func(x []float64) (float64, error) {
		// Example: Rosenbrock function
		return math.Pow(1-x[0], 2) + 100*math.Pow(x[1]-x[0]*x[0], 2), nil
	}

	// Define parameter bounds
	bounds := [][2]float64{{-5, 5}, {-5, 5}}

	// Create optimizer configuration
	config := optimization.OptimizerConfig{
		Objective:     objective,
		Bounds:        bounds,
		MaxIterations: 50,
		NInitialPoints: 10,
	}

	// Create a new Bayesian optimizer
	optimizer := bayesian.NewBayesianOptimizer(config)

	// Run the optimization
	result, err := optimizer.Optimize(context.Background())
	if err != nil {
		panic(fmt.Sprintf("Optimization failed: %v", err))
	}

	fmt.Printf("Optimal parameters: %v\n", result.X)
	fmt.Printf("Optimal value: %f\n", result.Y)
}
```

### Configuration

Create a `config.yaml` file to customize the server behavior:

```yaml
server:
  port: 8080
  env: development
  timeout: 30s

logging:
  level: info
  format: json
  output: stdout

optimization:
  max_concurrent: 4
  default_kernel: "matern52"
  default_acquisition: "ei"
  
storage:
  type: "memory"  # or "postgres"
  dsn: ""  # Only needed for postgres

metrics:
  enabled: true
  path: "/metrics"
  namespace: "tundr"
  
tracing:
  enabled: false
  service_name: "mcp-optimization-server"
  endpoint: "localhost:4317"
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out

# Run benchmarks
go test -bench=. -benchmem ./...
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest new features.

## 📄 License

This project is part of the CopyleftDev ecosystem and is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [Bayesian Optimization: A Tutorial](https://arxiv.org/abs/1807.02811)
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)
- [Model Context Protocol Specification](https://example.com/mcp-spec) (Coming Soon)

## 📬 Contact

For questions or support, please open an issue or contact the maintainers at [email protected]

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tundr/mcp-optimization-server.git
   cd mcp-optimization-server
   ```

2. Install dependencies:
   ```bash
   make deps
   ```

3. Build the binary:
   ```bash
   make build
   ```

   This will create a `tundr` binary in the `bin` directory.

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Application
ENV=development
LOG_LEVEL=info
HTTP_PORT=8080

# Database
DB_TYPE=sqlite  # sqlite or postgres
DB_DSN=file:data/tundr.db?cache=shared&_fk=1

# Authentication
JWT_KEY=your-secure-key-change-in-production

# Optimization
MAX_CONCURRENT_JOBS=10
JOB_TIMEOUT=30m

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

### Configuration File

For more complex configurations, you can use a YAML configuration file (default: `config/config.yaml`):

```yaml
server:
  env: development
  port: 8080
  shutdown_timeout: 30s

database:
  type: sqlite
  dsn: file:data/tundr.db?cache=shared&_fk=1
  max_open_conns: 25
  max_idle_conns: 5
  conn_max_lifetime: 5m

optimization:
  max_concurrent_jobs: 10
  job_timeout: 30m
  default_algorithm: bayesian
  
  bayesian:
    default_kernel: matern52
    default_noise: 1e-6
    max_observations: 1000
    
  cma_es:
    population_size: auto  # auto or number
    max_generations: 1000

monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
  
  tracing:
    enabled: false
    endpoint: localhost:4317
    sample_rate: 0.1

logging:
  level: info
  format: json
  enable_caller: true
  enable_stacktrace: true
```

## Running the Server

### Development Mode

For development with hot reload:

```bash
make dev
```

### Production Mode

Build and run the server:

```bash
make build
./bin/tundr serve --config config/production.yaml
```

### Using Docker

```bash
# Build the Docker image
docker build -t tundr-optimization .

# Run the container
docker run -p 8080:8080 -v $(pwd)/data:/app/data tundr-optimization
```

The server will be available at `http://localhost:8080`

## Usage Examples

### Bayesian Optimization Example

```go
package main

import (
	"context"
	"fmt"
	"log"
	"math"


	"github.com/tundr/mcp-optimization-server/internal/optimization"
	"github.com/tundr/mcp-optimization-server/internal/optimization/bayesian"
	"github.com/tundr/mcp-optimization-server/internal/optimization/kernels"
)

func main() {
	// Define the objective function (to be minimized)
	objective := func(x []float64) (float64, error) {
		// Example: Rosenbrock function
		return math.Pow(1-x[0], 2) + 100*math.Pow(x[1]-x[0]*x[0], 2), nil
	}

	// Define parameter bounds
	bounds := []optimization.Parameter{
		{Name: "x1", Min: -5.0, Max: 10.0},
		{Name: "x2", Min: -5.0, Max: 10.0},
	}

	// Create optimizer configuration
	config := optimization.OptimizerConfig{
		Objective:      objective,
		Parameters:     bounds,
		NInitialPoints: 10,
		MaxIterations:  50,
		Verbose:        true,
	}

	// Create and configure the optimizer
	optimizer, err := bayesian.NewBayesianOptimizer(config)
	if err != nil {
		log.Fatalf("Failed to create optimizer: %v", err)
	}

	// Run the optimization
	result, err := optimizer.Optimize(context.Background())
	if err != nil {
		log.Fatalf("Optimization failed: %v", err)
	}

	// Print results
	fmt.Printf("Best solution: %+v\n", result.BestSolution)
	fmt.Printf("Best value: %f\n", result.BestSolution.Value)
	fmt.Printf("Number of iterations: %d\n", len(result.History))
}
```

### REST API Example

Start a new optimization job:

```bash
curl -X POST http://localhost:8080/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "bayesian",
    "objective": "minimize",
    "parameters": [
      {"name": "x1", "type": "float", "bounds": [-5.0, 10.0]},
      {"name": "x2", "type": "float", "bounds": [-5.0, 10.0]}
    ],
    "max_iterations": 100,
    "n_initial_points": 20,
    "metadata": {
      "name": "rosenbrock-optimization",
      "tags": ["test", "demo"]
    }
  }'
```

Check optimization status:

```bash
curl http://localhost:8080/api/v1/status/<job_id>
```

## Configuration Reference

### Bayesian Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| kernel | string | "matern52" | Kernel type ("matern52" or "rbf") |
| length_scale | float | 1.0 | Length scale parameter |
| noise | float | 1e-6 | Observation noise |
| xi | float | 0.01 | Exploration-exploitation trade-off |
| n_initial_points | int | 10 | Number of initial random points |
| max_iterations | int | 100 | Maximum number of iterations |
| random_seed | int | 0 | Random seed (0 for time-based) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ENV | development | Application environment |
| LOG_LEVEL | info | Logging level |
| HTTP_PORT | 8080 | HTTP server port |
| DB_TYPE | sqlite | Database type (sqlite or postgres) |
| DB_DSN | file:data/tundr.db | Database connection string |
| JWT_KEY | | Secret key for JWT authentication |
| MAX_CONCURRENT_JOBS | 10 | Maximum concurrent optimization jobs |
| JOB_TIMEOUT | 30m | Maximum job duration |
| METRICS_ENABLED | true | Enable Prometheus metrics |
| METRICS_PORT | 9090 | Metrics server port |

## Advanced Usage

### Custom Kernels

You can implement custom kernel functions by implementing the `kernels.Kernel` interface:

```go
type Kernel interface {
    Eval(x, y []float64) float64
    Hyperparameters() []float64
    SetHyperparameters(params []float64) error
    Bounds() [][2]float64
}
```

Example custom kernel:

```go
type MyCustomKernel struct {
    lengthScale float64
    variance    float64
}

func (k *MyCustomKernel) Eval(x, y []float64) float64 {
    // Implement your custom kernel function
    sumSq := 0.0
    for i := range x {
        diff := x[i] - y[i]
        sumSq += diff * diff
    }
    return k.variance * math.Exp(-0.5*sumSq/(k.lengthScale*k.lengthScale))
}

// Implement other required methods...
```

### Parallel Evaluation

The optimizer supports parallel evaluation of multiple points:

```go
config := optimization.OptimizerConfig{
    Objective:      objective,
    Parameters:     bounds,
    NInitialPoints: 10,
    MaxIterations:  50,
    NJobs:         4,  // Use 4 parallel workers
}
```

### Callbacks

You can register callbacks to monitor the optimization process:

```go
optimizer := bayesian.NewBayesianOptimizer(config)

// Add a callback that's called after each iteration
optimizer.AddCallback(func(result *optimization.OptimizationResult) {
    fmt.Printf("Iteration %d: Best value = %f\n", 
        len(result.History), 
        result.BestSolution.Value)
})
```

## API Documentation

### REST API

#### Start Optimization

```
POST /api/v1/optimize
Content-Type: application/json

{
  "algorithm": "bayesian",
  "objective": "minimize",
  "parameters": [
    {"name": "x", "type": "float", "bounds": [0, 10], "prior": "uniform"},
    {"name": "y", "type": "float", "bounds": [-5, 5], "prior": "normal", "mu": 0, "sigma": 1}
  ],
  "constraints": [
    {"type": "ineq", "expr": "x + y <= 10"}
  ],
  "options": {
    "max_iterations": 100,
    "n_initial_points": 20,
    "acquisition": "ei",
    "xi": 0.01,
    "kappa": 0.1
  },
  "metadata": {
    "name": "example-optimization",
    "tags": ["test"],
    "user_id": "user123"
  }
}
```

#### Get Optimization Status

```
GET /api/v1/status/:id
```

Response:
```json
{
  "id": "job-123",
  "status": "running",
  "progress": 0.45,
  "best_solution": {
    "parameters": {"x": 1.2, "y": 3.4},
    "value": 0.123
  },
  "start_time": "2025-06-30T10:00:00Z",
  "elapsed_time": "1h23m45s",
  "iterations": 45,
  "metadata": {
    "name": "example-optimization",
    "tags": ["test"]
  }
}
```

### JSON-RPC 2.0 API

The server also supports JSON-RPC 2.0 for more advanced use cases:

```
POST /rpc
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "optimization.start",
  "params": [
    {
      "algorithm": "bayesian",
      "objective": "minimize",
      "parameters": [
        {"name": "x", "type": "float", "bounds": [0, 10]},
        {"name": "y", "type": "float", "bounds": [-5, 5]}
      ],
      "options": {
        "max_iterations": 100,
        "n_initial_points": 20,
        "acquisition": "ei",
        "xi": 0.01
      }
    }
  ]
}
```

## Performance Tuning

### Memory Usage

For large-scale problems, you may need to adjust the following parameters:

1. **Batch Size**: Process points in batches to limit memory usage
2. **GP Model**: Use a sparse approximation for large datasets (>1000 points)
3. **Cholesky Decomposition**: The default solver uses Cholesky decomposition with SVD fallback

### Parallelism

You can control the number of parallel workers:

```go
config := optimization.OptimizerConfig{
    // ... other options ...
    NJobs: runtime.NumCPU(),  // Use all available CPUs
}
```

### Caching

Enable caching of kernel matrix computations:

```go
kernel := kernels.NewMatern52Kernel(1.0, 1.0)
kernel.EnableCache(true)  // Enable kernel cache
```

## Monitoring and Observability

The server exposes Prometheus metrics at `/metrics`:

- `optimization_requests_total`: Total optimization requests
- `optimization_duration_seconds`: Duration of optimization jobs
- `optimization_iterations_total`: Number of iterations per optimization
- `optimization_errors_total`: Number of optimization errors
- `gp_fit_duration_seconds`: Duration of GP model fitting
- `acquisition_evaluations_total`: Number of acquisition function evaluations

### Logging

Logs are structured in JSON format by default. The following log levels are available:

- `debug`: Detailed debug information
- `info`: General operational information
- `warn`: Non-critical issues
- `error`: Critical errors

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Workflow

```bash
# Run tests
make test

# Run linters
make lint

# Run benchmarks
make benchmark

# Format code
make fmt

# Generate documentation
make docs
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Gonum](https://gonum.org/) - Numerical computing libraries for Go
- [Zap](https://github.com/uber-go/zap) - Blazing fast, structured, leveled logging
- [Chi](https://github.com/go-chi/chi) - Lightweight, composable router for Go HTTP services
- [Testify](https://github.com/stretchr/testify) - Toolkit with common assertions and mocks

## Development

### Building

```bash
make build
```

### Testing

```bash
make test
```

### Linting

```bash
make lint
```

## Deployment

### Docker

```bash
docker build -t tundr/mcp-optimization-server .
docker run -p 8080:8080 --env-file .env tundr/mcp-optimization-server
```

### Kubernetes

See the `deploy/kubernetes` directory for example Kubernetes manifests.

## License

[Apache License 2.0](LICENSE)
