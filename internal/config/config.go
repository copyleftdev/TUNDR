package config

import (
	"os"
	"strconv"
	"time"

	"github.com/caarlos0/env/v10"
)

type Config struct {
	Environment string `env:"ENV" envDefault:"development"`
	HTTP       struct {
		Port            int           `env:"HTTP_PORT" envDefault:"8080"`
		ReadTimeout     time.Duration `env:"HTTP_READ_TIMEOUT" envDefault:"30s"`
		WriteTimeout    time.Duration `env:"HTTP_WRITE_TIMEOUT" envDefault:"30s"`
		IdleTimeout     time.Duration `env:"HTTP_IDLE_TIMEOUT" envDefault:"120s"`
		ShutdownTimeout time.Duration `env:"HTTP_SHUTDOWN_TIMEOUT" envDefault:"30s"`
	}
	Logging struct {
		Level  string `env:"LOG_LEVEL" envDefault:"info"`
		Format string `env:"LOG_FORMAT" envDefault:"json"`
		Output string `env:"LOG_OUTPUT" envDefault:"stderr"`
	}
	Database struct {
		Type     string `env:"DB_TYPE" envDefault:"sqlite"`
		DSN      string `env:"DB_DSN" envDefault:"file:data/tundr.db?cache=shared&_fk=1"`
		MaxConns int    `env:"DB_MAX_CONNS" envDefault:"10"`
	}
	Auth struct {
		Enabled bool   `env:"AUTH_ENABLED" envDefault:"false"`
		JWTKey  string `env:"JWT_KEY,required"`
	}
	Optimization struct {
		WorkerCount int `env:"OPT_WORKER_COUNT" envDefault:"10"`
	}
}

func Load() (*Config, error) {
	cfg := &Config{}

	// Parse environment variables
	if err := env.Parse(cfg); err != nil {
		return nil, err
	}

	// Set default JWT key in development
	if cfg.Environment == "development" && !cfg.Auth.Enabled {
		cfg.Auth.JWTKey = "insecure-dev-key-change-in-production"
	}

	// Set default logging level based on environment
	if cfg.Environment == "development" && cfg.Logging.Level == "" {
		cfg.Logging.Level = "debug"
	}

	// Set default database DSN based on environment
	if cfg.Database.DSN == "" {
		switch cfg.Database.Type {
		case "sqlite":
			// Ensure the data directory exists
			if err := os.MkdirAll("data", 0755); err != nil {
				return nil, err
			}
			cfg.Database.DSN = "file:data/tundr.db?cache=shared&_fk=1"
		case "postgres":
			cfg.Database.DSN = "host=localhost port=5432 user=postgres password=postgres dbname=tundr sslmode=disable"
		}
	}

	return cfg, nil
}

// GetEnv returns the value of the environment variable or the default value
func GetEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

// GetEnvAsInt returns the value of the environment variable as int or the default value
func GetEnvAsInt(key string, defaultValue int) int {
	valueStr := GetEnv(key, "")
	if value, err := strconv.Atoi(valueStr); err == nil {
		return value
	}
	return defaultValue
}

// GetEnvAsBool returns the value of the environment variable as bool or the default value
func GetEnvAsBool(key string, defaultValue bool) bool {
	valueStr := GetEnv(key, "")
	if value, err := strconv.ParseBool(valueStr); err == nil {
		return value
	}
	return defaultValue
}
