# Contributing to TUNDR MCP Optimization Server

Thank you for your interest in contributing to TUNDR MCP Optimization Server! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone git@github.com:your-username/mcp-optimization.git
   cd mcp-optimization
   ```
3. Set up the development environment:
   ```bash
   # Install dependencies
   go mod download
   
   # Install development tools
   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
   go install github.com/vektra/mockery/v2@latest
   ```

## Development Workflow

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. Make your changes following the [code style](#code-style)

3. Run tests and linters:
   ```bash
   make test
   make lint
   ```

4. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "feat: add new optimization algorithm"
   git commit -m "fix: resolve issue with matrix inversion"
   ```

5. Push your branch and create a pull request

## Code Style

- Follow the [Effective Go](https://golang.org/doc/effective_go.html) guidelines
- Run `gofmt` and `goimports` on your code
- Keep functions small and focused
- Write clear and concise comments for exported types and functions
- Use descriptive variable and function names
- Handle errors explicitly
- Write tests for new functionality
- Document any MCP protocol extensions or modifications
- Follow the existing patterns for MCP endpoint implementations

## Testing

We aim for high test coverage. Please include tests with your contributions.

- Unit tests should be in the same package as the code being tested
- Table-driven tests are preferred for testing multiple scenarios
- Use testify/assert and require for assertions
- Benchmark tests are welcome for performance-critical code

To run tests:
```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out

# Run benchmarks
go test -bench=. -benchmem ./...
```

## Pull Request Process

1. Ensure your code passes all tests and linters
2. Update documentation as needed
3. Keep pull requests focused on a single feature or bug fix
4. Include tests that verify your changes
5. Update the CHANGELOG.md with a description of your changes
6. Request review from maintainers

## Reporting Issues

When reporting issues, please include:

- A clear title and description
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (Go version, OS, etc.)
- Any relevant error messages or logs

## Feature Requests

We welcome feature requests! Please open an issue with:

- A clear description of the feature
- The problem it solves
- Any relevant use cases
- Any alternative solutions you've considered

## Documentation

Good documentation is crucial. When making changes:

- Update relevant documentation
- Add examples for new features
- Keep API documentation up to date
- Ensure all exported types and functions have godoc comments

## MCP Compliance

When contributing to MCP-related functionality:

1. Follow the MCP specification for any protocol changes
2. Document any deviations from the standard
3. Include tests that verify MCP compliance
4. Update protocol documentation when adding new features

## License

By contributing, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0.
