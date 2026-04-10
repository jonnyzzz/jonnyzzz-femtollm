package health

import (
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Status tracks the health of a single backend.
type Status struct {
	Alive     bool
	LastCheck time.Time
	LastError string
}

// Checker periodically probes backends and tracks which are alive.
type Checker struct {
	mu       sync.RWMutex
	status   map[string]*Status // keyed by backend name
	backends []Backend
	client   *http.Client
	interval time.Duration
	stopCh   chan struct{}
}

// Backend is the minimal info needed for health checks.
type Backend struct {
	Name string
	URL  string
}

// NewChecker creates a health checker. Call Start() to begin background probing.
func NewChecker(backends []Backend, interval time.Duration, timeout time.Duration) *Checker {
	status := make(map[string]*Status, len(backends))
	for _, b := range backends {
		status[b.Name] = &Status{Alive: true} // fail-open: assume alive until first check
	}
	return &Checker{
		status:   status,
		backends: backends,
		client:   &http.Client{Timeout: timeout},
		interval: interval,
		stopCh:   make(chan struct{}),
	}
}

// Start runs an initial synchronous check, then launches background probing.
func (c *Checker) Start() {
	c.CheckNow()
	go c.loop()
}

// Stop signals the background goroutine to stop.
func (c *Checker) Stop() {
	close(c.stopCh)
}

// IsAlive returns whether the named backend is healthy.
// Returns true for unknown backends (fail-open).
func (c *Checker) IsAlive(name string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	s, ok := c.status[name]
	if !ok {
		return true
	}
	return s.Alive
}

// Statuses returns a snapshot of all backend health statuses.
func (c *Checker) Statuses() map[string]Status {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]Status, len(c.status))
	for k, v := range c.status {
		out[k] = *v
	}
	return out
}

// CheckNow runs a synchronous health check of all backends.
func (c *Checker) CheckNow() {
	var wg sync.WaitGroup
	for i := range c.backends {
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			c.checkOne(b)
		}(&c.backends[i])
	}
	wg.Wait()
}

func (c *Checker) loop() {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()
	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.CheckNow()
		}
	}
}

func (c *Checker) checkOne(b *Backend) {
	url := strings.TrimRight(b.URL, "/") + "/v1/models"
	resp, err := c.client.Get(url)

	c.mu.Lock()
	defer c.mu.Unlock()

	s := c.status[b.Name]
	if s == nil {
		s = &Status{}
		c.status[b.Name] = s
	}
	s.LastCheck = time.Now()

	if err != nil {
		wasAlive := s.Alive
		s.Alive = false
		s.LastError = err.Error()
		if wasAlive {
			log.Printf("health: backend %s is DOWN: %v", b.Name, err)
		}
		return
	}
	resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		wasAlive := s.Alive
		s.Alive = true
		s.LastError = ""
		if !wasAlive {
			log.Printf("health: backend %s is UP", b.Name)
		}
	} else {
		wasAlive := s.Alive
		s.Alive = false
		s.LastError = resp.Status
		if wasAlive {
			log.Printf("health: backend %s is DOWN: %s", b.Name, resp.Status)
		}
	}
}
