package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/protocol"
)

// newTestBackend creates a mock LLM backend server.
func newTestBackend(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	return httptest.NewServer(handler)
}

// chatCompletionHandler returns a handler that echoes the model and a fixed response.
func chatCompletionHandler(responseText string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]json.RawMessage
		_ = json.Unmarshal(body, &req)

		var model string
		_ = json.Unmarshal(req["model"], &model)

		stop := "stop"
		resp := protocol.ChatResponse{
			ID:    "test-123",
			Model: model,
			Choices: []protocol.ChatChoice{
				{
					Message:      protocol.ChatMessage{Role: "assistant", Content: mustMarshal(responseText)},
					FinishReason: &stop,
				},
			},
			Usage: &protocol.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

func mustMarshal(v any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}

func TestHealthEndpoint(t *testing.T) {
	cfg := &config.Config{Backends: []config.Backend{}}
	srv := NewServer(cfg)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestModelsEndpoint(t *testing.T) {
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Model: "Qwen/Qwen3"},
			{Name: "gpt", Model: "gpt-4"},
		},
	}
	srv := NewServer(cfg)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp protocol.ModelsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Errorf("expected 2 models, got %d", len(resp.Data))
	}
}

func TestChatCompletions_RoutesToBackend(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("Hello from vLLM"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "Qwen/Qwen3"},
		},
	}
	// Pre-compile
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp protocol.ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// Model should be rewritten to backend model
	if resp.Model != "Qwen/Qwen3" {
		t.Errorf("expected model Qwen/Qwen3, got %s", resp.Model)
	}
}

func TestChatCompletions_ModelRouting(t *testing.T) {
	backendA := newTestBackend(t, chatCompletionHandler("from A"))
	defer backendA.Close()
	backendB := newTestBackend(t, chatCompletionHandler("from B"))
	defer backendB.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Pattern: `(?i)qwen`, URL: backendA.URL, Model: "Qwen/Qwen3"},
			{Name: "fallback", Pattern: `.*`, URL: backendB.URL, Model: "default-model"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)

	// Request for qwen -> goes to backendA
	body := `{"model":"qwen3-coder","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	var resp protocol.ChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "Qwen/Qwen3" {
		t.Errorf("expected Qwen/Qwen3 for qwen request, got %s", resp.Model)
	}

	// Request for unknown model -> goes to fallback (backendB)
	body = `{"model":"llama-3","messages":[{"role":"user","content":"hi"}]}`
	req = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Model != "default-model" {
		t.Errorf("expected default-model for fallback, got %s", resp.Model)
	}
}

func TestChatCompletions_FallbackOnError(t *testing.T) {
	// First backend returns 500
	failingBackend := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"down"}`))
	})
	defer failingBackend.Close()

	// Second backend works
	workingBackend := newTestBackend(t, chatCompletionHandler("fallback worked"))
	defer workingBackend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "primary", Pattern: `.*`, URL: failingBackend.URL},
			{Name: "fallback", Pattern: `.*`, URL: workingBackend.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"any","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d: %s", w.Code, w.Body.String())
	}
}

func TestChatCompletions_NoBackend(t *testing.T) {
	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "qwen", Pattern: `qwen`, URL: "http://unused"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestAnthropicMessages_ConvertsToOpenAI(t *testing.T) {
	backend := newTestBackend(t, chatCompletionHandler("Converted response"))
	defer backend.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "vllm", Pattern: ".*", URL: backend.URL, Model: "Qwen/Qwen3"},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{
		"model": "claude-3-sonnet",
		"max_tokens": 1024,
		"system": "You are helpful",
		"messages": [{"role": "user", "content": "Hello"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp protocol.AnthropicResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Type != "message" {
		t.Errorf("expected type message, got %s", resp.Type)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", resp.StopReason)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "Converted response" {
		t.Errorf("expected converted response text, got %v", resp.Content)
	}
}

func TestAnthropicMessages_FallbackOnError(t *testing.T) {
	failing := newTestBackend(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer failing.Close()
	working := newTestBackend(t, chatCompletionHandler("ok"))
	defer working.Close()

	cfg := &config.Config{
		Backends: []config.Backend{
			{Name: "bad", Pattern: ".*", URL: failing.URL},
			{Name: "good", Pattern: ".*", URL: working.URL},
		},
	}
	for i := range cfg.Backends {
		cfg.Backends[i].Match("test")
	}

	srv := NewServer(cfg)
	body := `{"model":"x","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d", w.Code)
	}
}
