package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/config"
	"github.com/jonnyzzz/jonnyzzz-femtollm/internal/protocol"
)

// Server is the LLM proxy server.
type Server struct {
	Config *config.Config
	Client *http.Client
}

// NewServer creates a new proxy server.
func NewServer(cfg *config.Config) *Server {
	return &Server{
		Config: cfg,
		Client: &http.Client{Timeout: 5 * time.Minute},
	}
}

// Handler returns the HTTP handler for the proxy.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// OpenAI-compatible endpoints
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)

	// Anthropic-compatible endpoint
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)

	// Health
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	return mux
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	models := s.Config.AdvertisedModels()
	var entries []protocol.ModelEntry
	for _, m := range models {
		entries = append(entries, protocol.ModelEntry{
			ID:      m,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "femtollm",
		})
	}

	resp := protocol.ModelsResponse{Object: "list", Data: entries}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req protocol.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	backends := s.Config.FindBackends(req.Model)
	if len(backends) == 0 {
		writeError(w, http.StatusNotFound, fmt.Sprintf("no backend for model %q", req.Model))
		return
	}

	// Try backends in order (fallback)
	var lastErr error
	for _, backend := range backends {
		err := s.forwardOpenAI(w, r, body, &req, &backend)
		if err == nil {
			return
		}
		lastErr = err
		log.Printf("Backend %s failed for model %s: %v", backend.Name, req.Model, err)
	}

	writeError(w, http.StatusBadGateway, fmt.Sprintf("all backends failed: %v", lastErr))
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req protocol.AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	if req.Stream {
		// For streaming, convert to OpenAI and stream back
		// TODO: implement streaming protocol conversion
		writeError(w, http.StatusNotImplemented, "anthropic streaming not yet supported, use non-streaming")
		return
	}

	backends := s.Config.FindBackends(req.Model)
	if len(backends) == 0 {
		writeError(w, http.StatusNotFound, fmt.Sprintf("no backend for model %q", req.Model))
		return
	}

	// Convert to OpenAI format, send to backend, convert response back
	openaiReq := protocol.AnthropicToOpenAI(&req)

	var lastErr error
	for _, backend := range backends {
		openaiReq.Model = backend.TargetModel(req.Model)

		reqBody, _ := json.Marshal(openaiReq)
		url := strings.TrimRight(backend.URL, "/") + "/v1/chat/completions"

		backendReq, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(reqBody))
		backendReq.Header.Set("Content-Type", "application/json")
		if backend.APIKey != "" {
			backendReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
		}

		resp, err := s.Client.Do(backendReq)
		if err != nil {
			lastErr = fmt.Errorf("backend %s: %w", backend.Name, err)
			log.Printf("Backend %s failed: %v", backend.Name, err)
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("backend %s: read response: %w", backend.Name, err)
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("backend %s: status %d", backend.Name, resp.StatusCode)
			log.Printf("Backend %s returned %d, trying next", backend.Name, resp.StatusCode)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.StatusCode)
			_, _ = w.Write(respBody)
			return
		}

		var openaiResp protocol.ChatResponse
		if err := json.Unmarshal(respBody, &openaiResp); err != nil {
			writeError(w, http.StatusBadGateway, "invalid backend response")
			return
		}

		anthropicResp := protocol.OpenAIToAnthropicResponse(&openaiResp)
		writeJSON(w, http.StatusOK, anthropicResp)
		return
	}

	writeError(w, http.StatusBadGateway, fmt.Sprintf("all backends failed: %v", lastErr))
}

// forwardOpenAI proxies a request to an OpenAI-compatible backend.
// For streaming requests, it pipes the SSE response directly.
func (s *Server) forwardOpenAI(w http.ResponseWriter, r *http.Request, body []byte, req *protocol.ChatRequest, backend *config.Backend) error {
	// Rewrite model name
	var modified map[string]json.RawMessage
	if err := json.Unmarshal(body, &modified); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	targetModel := backend.TargetModel(req.Model)
	modified["model"], _ = json.Marshal(targetModel)
	reqBody, _ := json.Marshal(modified)

	url := strings.TrimRight(backend.URL, "/") + "/v1/chat/completions"
	backendReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	backendReq.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		backendReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}

	// Use a client without timeout for streaming
	client := s.Client
	if req.Stream {
		client = &http.Client{}
	}

	resp, err := client.Do(backendReq)
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}

	if resp.StatusCode >= 500 {
		resp.Body.Close()
		return fmt.Errorf("status %d", resp.StatusCode)
	}

	// Pipe response directly (works for both streaming and non-streaming)
	for k, vals := range resp.Header {
		for _, v := range vals {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
	resp.Body.Close()
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "proxy_error",
		},
	})
}
