package logging

import (
	"io"
	"os"
	"strings"
)

// Config holds the configuration for the logger.
type Config struct {
	// Level is the minimum log level to output (DEBUG, INFO, WARN, ERROR, FATAL)
	Level string `yaml:"level"`
	// Format is the output format (json, text)
	Format string `yaml:"format"`
	// Output is the output destination (stdout, stderr, or file path)
	Output string `yaml:"output"`
}

// DefaultConfig returns the default logging configuration.
func DefaultConfig() *Config {
	return &Config{
		Level:  "info",
		Format: "json",
		Output: "stderr",
	}
}

// NewLogger creates a new logger with the given configuration.
func NewLogger(cfg *Config) (*Logger, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// Parse log level
	level := parseLevel(cfg.Level)

	// Get output writer
	output, err := getOutput(cfg.Output)
	if err != nil {
		return nil, err
	}

	return New(level, output), nil
}

// parseLevel converts a string log level to LogLevel.
func parseLevel(level string) LogLevel {
	switch strings.ToUpper(level) {
	case "DEBUG":
		return DebugLevel
	case "INFO":
		return InfoLevel
	case "WARN":
		return WarnLevel
	case "ERROR":
		return ErrorLevel
	case "FATAL":
		return FatalLevel
	default:
		return InfoLevel
	}
}

// getOutput returns an io.Writer for the given output destination.
func getOutput(output string) (io.Writer, error) {
	switch output {
	case "stdout":
		return os.Stdout, nil
	case "stderr":
		return os.Stderr, nil
	default:
		// Treat as file path
		file, err := os.OpenFile(output, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return nil, err
		}
		return file, nil
	}
}
