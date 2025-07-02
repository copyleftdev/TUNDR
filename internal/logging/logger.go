// Package logging provides structured logging for the TUNDR MCP Optimization Server.
package logging

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"
)

// LogLevel represents the severity level of a log entry.
type LogLevel string

const (
	// DebugLevel logs are typically voluminous, and are usually disabled in
	// production.
	DebugLevel LogLevel = "DEBUG"
	// InfoLevel is the default logging priority.
	InfoLevel LogLevel = "INFO"
	// WarnLevel logs are more important than Info, but don't need individual
	// human review.
	WarnLevel LogLevel = "WARN"
	// ErrorLevel logs are high-priority. If an application is running smoothly,
	// it shouldn't generate any error-level logs.
	ErrorLevel LogLevel = "ERROR"
	// FatalLevel logs a message, then calls os.Exit(1).
	FatalLevel LogLevel = "FATAL"
)

// Logger represents an active logging object.
type Logger struct {
	level  LogLevel
	output io.Writer
	fields map[string]interface{}
}

// New creates a new Logger with the specified log level and output.
func New(level LogLevel, output io.Writer) *Logger {
	return &Logger{
		level:  level,
		output: output,
		fields: make(map[string]interface{}),
	}
}

// WithFields returns a new Logger with the specified fields.
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	newFields := make(map[string]interface{}, len(l.fields)+len(fields))
	for k, v := range l.fields {
		newFields[k] = v
	}
	for k, v := range fields {
		newFields[k] = v
	}

	return &Logger{
		level:  l.level,
		output: l.output,
		fields: newFields,
	}
}

// WithField returns a new Logger with the specified key-value pair.
func (l *Logger) WithField(key string, value interface{}) *Logger {
	return l.WithFields(map[string]interface{}{key: value})
}

// WithError returns a new Logger with the error field set.
func (l *Logger) WithError(err error) *Logger {
	return l.WithField("error", err.Error())
}

// log writes a log entry with the given level and message.
func (l *Logger) log(level LogLevel, msg string, fields map[string]interface{}) {
	if !l.shouldLog(level) {
		return
	}

	// Get caller info
	_, file, line, ok := runtime.Caller(2) // Adjust the depth as needed
	if !ok {
		file = "???"
		line = 0
	} else {
		// Only keep the last two parts of the file path
		parts := strings.Split(file, "/")
		if len(parts) > 2 {
			file = strings.Join(parts[len(parts)-2:], "/")
		}
	}

	// Merge fields
	allFields := make(map[string]interface{})
	for k, v := range l.fields {
		allFields[k] = v
	}
	for k, v := range fields {
		allFields[k] = v
	}

	// Create log entry
	entry := map[string]interface{}{
		"timestamp": time.Now().UTC().Format(time.RFC3339Nano),
		"level":     level,
		"message":   msg,
		"caller":    fmt.Sprintf("%s:%d", file, line),
	}

	// Add fields to entry
	for k, v := range allFields {
		entry[k] = v
	}

	// Encode to JSON
	jsonData, err := json.Marshal(entry)
	if err != nil {
		// Fallback to simple log if JSON encoding fails
		fmt.Fprintf(l.output, "%s [%s] %s: %+v\n", 
			time.Now().Format(time.RFC3339), level, msg, allFields)
		return
	}

	// Write to output
	jsonData = append(jsonData, '\n')
	_, _ = l.output.Write(jsonData)

	// Handle fatal level
	if level == FatalLevel {
		os.Exit(1)
	}
}

// shouldLog returns true if the given level should be logged.
func (l *Logger) shouldLog(level LogLevel) bool {
	var levels = map[LogLevel]int{
		DebugLevel: 0,
		InfoLevel:  1,
		WarnLevel:  2,
		ErrorLevel: 3,
		FatalLevel: 4,
	}

	shouldLog, exists := levels[level]
	if !exists {
		return false
	}

	currentLevel, exists := levels[l.level]
	if !exists {
		return false
	}

	return shouldLog >= currentLevel
}

// Debug logs a message at DebugLevel.
func (l *Logger) Debug(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	l.log(DebugLevel, msg, f)
}

// Info logs a message at InfoLevel.
func (l *Logger) Info(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	l.log(InfoLevel, msg, f)
}

// Warn logs a message at WarnLevel.
func (l *Logger) Warn(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	l.log(WarnLevel, msg, f)
}

// Error logs a message at ErrorLevel.
func (l *Logger) Error(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	l.log(ErrorLevel, msg, f)
}

// Fatal logs a message at FatalLevel then calls os.Exit(1).
func (l *Logger) Fatal(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	l.log(FatalLevel, msg, f)
}

// CtxLogger is a logger that can be used with context.
type CtxLogger struct {
	*Logger
}

// FromContext returns a logger from the context or a new one if none exists.
func FromContext(ctx context.Context) *CtxLogger {
	if logger, ok := ctx.Value(ctxLoggerKey{}).(*CtxLogger); ok {
		return logger
	}
	return &CtxLogger{New(InfoLevel, os.Stderr)}
}

// WithContext returns a new context with the logger.
func (l *CtxLogger) WithContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, ctxLoggerKey{}, l)
}

type ctxLoggerKey struct{}
