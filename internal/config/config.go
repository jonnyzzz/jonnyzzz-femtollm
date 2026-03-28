package config

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
)

// Backend defines an LLM backend with routing rules.
type Backend struct {
	Name     string `json:"name"`
	Pattern  string `json:"pattern"`            // regex to match model names
	URL      string `json:"url"`                // backend base URL (e.g. http://localhost:8000)
	Model    string `json:"model,omitempty"`     // override model name sent to backend
	APIKey   string `json:"api_key,omitempty"`   // API key for the backend
	Protocol string `json:"protocol,omitempty"`  // "openai" (default), "anthropic"
	Priority int    `json:"priority,omitempty"`  // lower = tried first for fallback groups

	compiled *regexp.Regexp
}

// Match returns true if the model name matches this backend's pattern.
func (b *Backend) Match(model string) bool {
	if b.compiled == nil {
		b.compiled = regexp.MustCompile(b.Pattern)
	}
	return b.compiled.MatchString(model)
}

// TargetModel returns the model name to use when forwarding to this backend.
func (b *Backend) TargetModel(requestModel string) string {
	if b.Model != "" {
		return b.Model
	}
	return requestModel
}

// Config holds the proxy configuration.
type Config struct {
	Listen   string    `json:"listen"`   // address to listen on (default ":8080")
	Backends []Backend `json:"backends"` // ordered list of backends
}

// Load reads configuration from a JSON file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if cfg.Listen == "" {
		cfg.Listen = ":8080"
	}
	// Pre-compile regexes
	for i := range cfg.Backends {
		re, err := regexp.Compile(cfg.Backends[i].Pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid pattern %q for backend %q: %w", cfg.Backends[i].Pattern, cfg.Backends[i].Name, err)
		}
		cfg.Backends[i].compiled = re
	}
	return &cfg, nil
}

// FindBackends returns all backends matching the given model, ordered by priority.
func (c *Config) FindBackends(model string) []Backend {
	var matches []Backend
	for _, b := range c.Backends {
		if b.Match(model) {
			matches = append(matches, b)
		}
	}
	return matches
}

// AdvertisedModels returns the deduplicated list of model names from all backends.
func (c *Config) AdvertisedModels() []string {
	seen := map[string]bool{}
	var models []string
	for _, b := range c.Backends {
		name := b.Model
		if name == "" {
			name = b.Name
		}
		if !seen[name] {
			seen[name] = true
			models = append(models, name)
		}
	}
	return models
}
