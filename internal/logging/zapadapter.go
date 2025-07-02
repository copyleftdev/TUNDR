package logging

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// ZapAdapter wraps our Logger to implement the zapcore.Core interface
type ZapAdapter struct {
	logger *Logger
}

// NewZapAdapter creates a new zapcore.Core that forwards logs to our Logger
func NewZapAdapter(logger *Logger) *ZapAdapter {
	return &ZapAdapter{
		logger: logger,
	}
}

// Enabled implements zapcore.Core
func (a *ZapAdapter) Enabled(level zapcore.Level) bool {
	var lvl LogLevel

	switch level {
	case zapcore.DebugLevel:
		lvl = DebugLevel
	case zapcore.InfoLevel:
		lvl = InfoLevel
	case zapcore.WarnLevel:
		lvl = WarnLevel
	case zapcore.ErrorLevel, zapcore.DPanicLevel, zapcore.PanicLevel, zapcore.FatalLevel:
		lvl = ErrorLevel
	default:
		lvl = InfoLevel
	}

	return a.logger.shouldLog(lvl)
}

// getFieldValue converts a zapcore.Field to its interface{} value
func getFieldValue(field zapcore.Field) interface{} {
	switch field.Type {
	case zapcore.StringType:
		return field.String
	case zapcore.Int64Type, zapcore.Int32Type, zapcore.Int16Type, zapcore.Int8Type:
		return field.Integer
	case zapcore.Float64Type, zapcore.Float32Type:
		return *(*float64)(field.Interface.(*float64))
	case zapcore.BoolType:
		return field.Integer == 1
	case zapcore.ErrorType:
		return field.Interface
	default:
		return field.Interface
	}
}

// With implements zapcore.Core
func (a *ZapAdapter) With(fields []zapcore.Field) zapcore.Core {
	// Convert zap fields to our fields format
	f := make(map[string]interface{}, len(fields))
	for _, field := range fields {
		f[field.Key] = getFieldValue(field)
	}

	return &ZapAdapter{
		logger: a.logger.WithFields(f),
	}
}

// Check implements zapcore.Core
func (a *ZapAdapter) Check(ent zapcore.Entry, ce *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	if a.Enabled(ent.Level) {
		return ce.AddCore(ent, a)
	}
	return ce
}

// Write implements zapcore.Core
func (a *ZapAdapter) Write(ent zapcore.Entry, fields []zapcore.Field) error {
	// Convert fields to our format
	f := make(map[string]interface{}, len(fields)+1)
	f["caller"] = ent.Caller.String()

	for _, field := range fields {
		f[field.Key] = getFieldValue(field)
	}

	// Map zap levels to our levels
	var lvl LogLevel

	switch ent.Level {
	case zapcore.DebugLevel:
		lvl = DebugLevel
	case zapcore.InfoLevel:
		lvl = InfoLevel
	case zapcore.WarnLevel:
		lvl = WarnLevel
	case zapcore.ErrorLevel, zapcore.DPanicLevel, zapcore.PanicLevel, zapcore.FatalLevel:
		lvl = ErrorLevel
	default:
		lvl = InfoLevel
	}

	// Write the log
	a.logger.log(lvl, ent.Message, f)

	// Handle fatal level
	if ent.Level == zapcore.FatalLevel {
		a.logger.Fatal(ent.Message, f)
	}

	return nil
}

// Sync implements zapcore.Core
func (a *ZapAdapter) Sync() error {
	// No-op for our logger
	return nil
}

// NewZapLogger creates a new *zap.Logger that forwards logs to our Logger
func NewZapLogger(logger *Logger) *zap.Logger {
	core := NewZapAdapter(logger)
	return zap.New(core)
}
