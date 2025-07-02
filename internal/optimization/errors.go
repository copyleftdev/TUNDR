package optimization

import "fmt"

// Error represents an optimization error with context
// that can be wrapped with additional information.
type Error struct {
	// Message describes the error that occurred.
	Message string
	// Op is the operation that caused the error.
	Op string
	// Component is the component where the error occurred.
	Component string
	// Err is the underlying error that triggered this one, if any.
	Err error
}

// Error returns the string representation of the error.
func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	var prefix string
	if e.Component != "" && e.Op != "" {
		prefix = fmt.Sprintf("%s: %s", e.Component, e.Op)
	} else if e.Component != "" {
		prefix = e.Component
	} else if e.Op != "" {
		prefix = e.Op
	}

	if e.Err != nil {
		if prefix != "" {
			return fmt.Sprintf("%s: %s: %v", prefix, e.Message, e.Err)
		}
		return fmt.Sprintf("%s: %v", e.Message, e.Err)
	}

	if prefix != "" {
		return fmt.Sprintf("%s: %s", prefix, e.Message)
	}
	return e.Message
}

// Unwrap returns the underlying error, if any.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// WithOperation adds operation context to the error.
func (e *Error) WithOperation(op string) *Error {
	e.Op = op
	return e
}

// WithComponent adds component context to the error.
func (e *Error) WithComponent(component string) *Error {
	e.Component = component
	return e
}

// NewError creates a new optimization error with the given message.
// The op parameter describes the operation that caused the error.
func NewError(message string) *Error {
	return &Error{
		Message: message,
	}
}

// NewErrorf creates a new optimization error with formatted message.
func NewErrorf(format string, args ...interface{}) *Error {
	return &Error{
		Message: fmt.Sprintf(format, args...),
	}
}

// WrapError wraps an existing error with additional context.
// The op parameter describes the operation that caused the error.
// If err is nil, WrapError returns nil.
func WrapError(err error, message string) *Error {
	if err == nil {
		return nil
	}
	return &Error{
		Message: message,
		Err:     err,
	}
}

// WrapErrorf wraps an existing error with additional formatted context.
// If err is nil, WrapErrorf returns nil.
func WrapErrorf(err error, format string, args ...interface{}) *Error {
	if err == nil {
		return nil
	}
	return &Error{
		Message: fmt.Sprintf(format, args...),
		Err:     err,
	}
}

// IsOptimizationError checks if an error is of type Error.
// If the error is an optimization error, it returns the error and true.
// Otherwise, it returns nil and false.
func IsOptimizationError(err error) (*Error, bool) {
	if err == nil {
		return nil, false
	}
	if e, ok := err.(*Error); ok {
		return e, true
	}
	return nil, false
}
