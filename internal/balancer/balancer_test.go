package balancer

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/health"
)

func backends(names ...string) []config.Backend {
	var out []config.Backend
	for _, n := range names {
		out = append(out, config.Backend{Name: n, URL: "http://" + n})
	}
	return out
}

func TestSelect_RoundRobin(t *testing.T) {
	b := NewBalancer(nil) // nil checker = all healthy
	bs := backends("a", "b", "c")

	counts := map[string]int{}
	for i := 0; i < 9; i++ {
		selected := b.Select(bs, "model")
		counts[selected[0].Name]++
	}

	for _, name := range []string{"a", "b", "c"} {
		if counts[name] != 3 {
			t.Errorf("expected backend %s to start first 3 times, got %d", name, counts[name])
		}
	}
}

func TestSelect_FiltersUnhealthy(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthy.Close()
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := dead.URL
	dead.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "alive", URL: healthy.URL},
		{Name: "dead", URL: deadURL},
	}, time.Hour, 1*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := []config.Backend{
		{Name: "alive", URL: healthy.URL},
		{Name: "dead", URL: deadURL},
	}

	selected := b.Select(bs, "model")
	if len(selected) != 1 {
		t.Fatalf("expected 1 healthy backend, got %d", len(selected))
	}
	if selected[0].Name != "alive" {
		t.Errorf("expected alive backend, got %s", selected[0].Name)
	}
}

func TestSelect_AllDead_FailOpen(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := srv.URL
	srv.Close()

	checker := health.NewChecker([]health.Backend{
		{Name: "a", URL: deadURL},
		{Name: "b", URL: deadURL},
	}, time.Hour, 1*time.Second)
	checker.CheckNow()

	b := NewBalancer(checker)
	bs := backends("a", "b")

	selected := b.Select(bs, "model")
	if len(selected) != 2 {
		t.Fatalf("expected all 2 backends on fail-open, got %d", len(selected))
	}
}

func TestSelect_SingleBackend(t *testing.T) {
	b := NewBalancer(nil)
	bs := backends("only")

	selected := b.Select(bs, "model")
	if len(selected) != 1 || selected[0].Name != "only" {
		t.Errorf("single backend should pass through unchanged")
	}
}

func TestSelect_Empty(t *testing.T) {
	b := NewBalancer(nil)
	selected := b.Select(nil, "model")
	if len(selected) != 0 {
		t.Error("expected empty result for empty input")
	}
}
