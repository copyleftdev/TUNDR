# Changelog

All notable changes to the CopyleftDev MCP Optimization Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Updated repository references to copyleftdev organization
- Updated license to AGPL-3.0
- Enhanced documentation with MCP protocol details

### Added
- Initial implementation of Bayesian Optimization with Gaussian Processes
- Support for multiple kernel functions (Matern 5/2, RBF)
- Expected Improvement acquisition function
- Comprehensive test suite with high coverage
- Basic command-line interface
- Docker support
- Prometheus metrics endpoint
- Structured logging with zap

### Changed
- Improved numerical stability in matrix operations
- Enhanced error handling and validation
- Optimized memory usage with object pooling
- Refactored API for better extensibility

### Fixed
- Fixed dimension mismatches in test assertions
- Resolved edge cases in GP predictions
- Addressed race conditions in concurrent operations
- Fixed error message formatting and consistency

## [0.1.0] - 2025-07-02

Initial release as part of the CopyleftDev ecosystem

### Added
- Initial release of TUNDR MCP Optimization Server
- Core Bayesian optimization functionality
- Basic API endpoints for optimization tasks
- Documentation and examples

### Changed
- N/A (Initial release)

### Fixed
- N/A (Initial release)
