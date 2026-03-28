package protocol

import "encoding/json"

// Anthropic Messages API request/response types.

type AnthropicRequest struct {
	Model       string            `json:"model"`
	Messages    []AnthropicMsg    `json:"messages"`
	System      json.RawMessage   `json:"system,omitempty"` // string or array
	Stream      bool              `json:"stream,omitempty"`
	MaxTokens   int               `json:"max_tokens"`
	Temperature *float64          `json:"temperature,omitempty"`
	Tools       json.RawMessage   `json:"tools,omitempty"`
}

type AnthropicMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or array of content blocks
}

type AnthropicResponse struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"` // "message"
	Role       string             `json:"role"`
	Model      string             `json:"model"`
	Content    []AnthropicContent `json:"content"`
	StopReason string             `json:"stop_reason,omitempty"`
	Usage      *AnthropicUsage    `json:"usage,omitempty"`
}

type AnthropicContent struct {
	Type  string          `json:"type"` // "text", "tool_use"
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicToOpenAI converts an Anthropic request to OpenAI Chat Completions format.
func AnthropicToOpenAI(req *AnthropicRequest) *ChatRequest {
	var messages []ChatMessage

	// System prompt
	if req.System != nil {
		var sysText string
		if err := json.Unmarshal(req.System, &sysText); err != nil {
			// Array format — use raw
			sysText = string(req.System)
		}
		if sysText != "" {
			messages = append(messages, ChatMessage{
				Role:    "system",
				Content: mustMarshal(sysText),
			})
		}
	}

	// Convert messages
	for _, msg := range req.Messages {
		messages = append(messages, ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	cr := &ChatRequest{
		Model:       req.Model,
		Messages:    messages,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		Tools:       req.Tools,
	}
	if req.MaxTokens > 0 {
		cr.MaxTokens = &req.MaxTokens
	}
	return cr
}

// OpenAIToAnthropicResponse converts an OpenAI response to Anthropic format.
func OpenAIToAnthropicResponse(resp *ChatResponse) *AnthropicResponse {
	ar := &AnthropicResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: resp.Model,
	}

	if resp.Usage != nil {
		ar.Usage = &AnthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
	}

	for _, choice := range resp.Choices {
		msg := choice.Message
		// Text content
		var textContent string
		if msg.Content != nil {
			_ = json.Unmarshal(msg.Content, &textContent)
		}
		if textContent != "" {
			ar.Content = append(ar.Content, AnthropicContent{
				Type: "text",
				Text: textContent,
			})
		}

		if choice.FinishReason != nil {
			switch *choice.FinishReason {
			case "tool_calls":
				ar.StopReason = "tool_use"
			case "stop":
				ar.StopReason = "end_turn"
			case "length":
				ar.StopReason = "max_tokens"
			default:
				ar.StopReason = *choice.FinishReason
			}
		}
	}

	return ar
}

func mustMarshal(v any) json.RawMessage {
	data, _ := json.Marshal(v)
	return data
}
