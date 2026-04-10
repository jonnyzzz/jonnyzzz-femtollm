package health

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestChecker_HealthyBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "ok", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	if !c.IsAlive("ok") {
		t.Error("expected backend to be alive")
	}
}

func TestChecker_UnhealthyBackend(t *testing.T) {
	// Use a closed server to simulate unreachable backend
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()

	c := NewChecker([]Backend{{Name: "down", URL: url}}, time.Hour, 1*time.Second)
	c.CheckNow()

	if c.IsAlive("down") {
		t.Error("expected backend to be dead")
	}
}

func TestChecker_500IsUnhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "err", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	if c.IsAlive("err") {
		t.Error("expected 500 backend to be unhealthy")
	}
}

func TestChecker_Recovery(t *testing.T) {
	alive := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if alive {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "b", URL: srv.URL}}, time.Hour, 5*time.Second)

	// Initially alive
	c.CheckNow()
	if !c.IsAlive("b") {
		t.Fatal("expected alive initially")
	}

	// Goes down
	alive = false
	c.CheckNow()
	if c.IsAlive("b") {
		t.Fatal("expected dead after failure")
	}

	// Comes back
	alive = true
	c.CheckNow()
	if !c.IsAlive("b") {
		t.Fatal("expected alive after recovery")
	}
}

func TestChecker_UnknownBackendIsAlive(t *testing.T) {
	c := NewChecker(nil, time.Hour, 5*time.Second)
	if !c.IsAlive("nonexistent") {
		t.Error("expected unknown backend to be treated as alive (fail-open)")
	}
}

func TestChecker_Statuses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := NewChecker([]Backend{{Name: "a", URL: srv.URL}}, time.Hour, 5*time.Second)
	c.CheckNow()

	statuses := c.Statuses()
	s, ok := statuses["a"]
	if !ok {
		t.Fatal("expected status for backend 'a'")
	}
	if !s.Alive {
		t.Error("expected alive")
	}
	if s.LastCheck.IsZero() {
		t.Error("expected LastCheck to be set")
	}
}

func TestChecker_StopGraceful(t *testing.T) {
	c := NewChecker(nil, 10*time.Millisecond, 5*time.Second)
	c.Start()
	c.Stop()
	// Should not panic or hang
}
