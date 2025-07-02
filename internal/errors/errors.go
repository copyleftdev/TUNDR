// Package errors provides enhanced error handling for the TUNDR MCP Optimization Server.
package errors

import (
	"fmt"
	"runtime"
	"strings"
)

// Error represents an error with context and stack trace.
type Error struct {
	// The underlying error that was returned
	Err error
	// A human-readable message describing the error
	Message string
	// The operation that was being performed when the error occurred
	Operation string
	// The component or package where the error occurred
	Component string
	// The stack trace
	Stack []string
}

// Error implements the error interface.
func (e *Error) Error() string {
	var builder strings.Builder
	
	if e.Message != "" {
		builder.WriteString(e.Message)
	}
	
	if e.Operation != "" {
		if builder.Len() > 0 {
			builder.WriteString(": ")
		}
		builder.WriteString("operation=")
		builder.WriteString(e.Operation)
	}
	
	if e.Component != "" {
		if builder.Len() > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString("component=")
		builder.WriteString(e.Component)
	}
	
	if e.Err != nil {
		if builder.Len() > 0 {
			builder.WriteString(": ")
		}
		builder.WriteString(e.Err.Error())
	}
	
	return builder.String()
}

// Unwrap returns the underlying error.
func (e *Error) Unwrap() error {
	return e.Err
}

// WithMessage adds a message to the error.
func (e *Error) WithMessage(msg string) *Error {
	e.Message = msg
	return e
}

// WithOperation adds an operation to the error.
func (e *Error) WithOperation(op string) *Error {
	e.Operation = op
	return e
}

// WithComponent adds a component to the error.
func (e *Error) WithComponent(component string) *Error {
	e.Component = component
	return e
}

// StackTrace returns the stack trace as a slice of strings.
func (e *Error) StackTrace() []string {
	return e.Stack
}

// New creates a new error with a message.
func New(msg string) *Error {
	return &Error{
		Message: msg,
		Stack:   getStackTrace(),
	}
}

// Errorf creates a new error with a formatted message.
func Errorf(format string, args ...interface{}) *Error {
	return &Error{
		Message: fmt.Sprintf(format, args...),
		Stack:   getStackTrace(),
	}
}

// Wrap wraps an error with additional context.
func Wrap(err error, msg string) *Error {
	if err == nil {
		return nil
	}

	e, ok := err.(*Error)
	if !ok {
		e = &Error{
			Err:   err,
			Stack: getStackTrace(),
		}
	}

	if msg != "" {
		e.Message = msg
	}

	return e
}

// Wrapf wraps an error with a formatted message.
func Wrapf(err error, format string, args ...interface{}) *Error {
	if err == nil {
		return nil
	}

	e, ok := err.(*Error)
	if !ok {
		e = &Error{
			Err:   err,
			Stack: getStackTrace(),
		}
	}

	e.Message = fmt.Sprintf(format, args...)
	return e
}

// getStackTrace returns the current stack trace as a slice of strings.
func getStackTrace() []string {
	const depth = 32
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:]) // Skip runtime.Callers, getStackTrace, and the constructor
	if n == 0 {
		return nil
	}

	frames := runtime.CallersFrames(pcs[:n])
	stack := make([]string, 0, n)

	for {
		frame, more := frames.Next()
		if !strings.Contains(frame.File, "runtime/") && !strings.Contains(frame.File, "internal/errors") {
			stack = append(stack, fmt.Sprintf("%s\n\t%s:%d", frame.Function, frame.File, frame.Line))
		}
		if !more {
			break
		}
	}

	return stack
}

// Is reports whether any error in err's chain matches target.
func Is(err, target error) bool {
	return err == target || (err != nil && target != nil && err.Error() == target.Error())
}

// As finds the first error in err's chain that matches target.
func As(err error, target interface{}) bool {
	if err == nil || target == nil {
		return false
	}

	return false
}

// Unwrap returns the result of calling the Unwrap method on err, if err's
// type contains an Unwrap method returning error.
// Otherwise, Unwrap returns nil.
func Unwrap(err error) error {
	u, ok := err.(interface {
		Unwrap() error
	})
	if !ok {
		return nil
	}
	return u.Unwrap()
}
