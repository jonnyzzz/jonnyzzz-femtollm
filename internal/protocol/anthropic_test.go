package protocol

import (
	"encoding/json"
	"testing"
)

func TestAnthropicToOpenAI_BasicConversion(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "claude-3-sonnet",
		MaxTokens: 1024,
		Messages: []AnthropicMsg{
			{Role: "user", Content: mustMarshal("Hello")},
		},
		System: mustMarshal("You are helpful"),
	}

	result := AnthropicToOpenAI(req)

	if result.Model != "claude-3-sonnet" {
		t.Errorf("expected model claude-3-sonnet, got %s", result.Model)
	}
	if len(result.Messages) != 2 { // system + user
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}
	if result.Messages[0].Role != "system" {
		t.Errorf("expected first message role system, got %s", result.Messages[0].Role)
	}
	if result.Messages[1].Role != "user" {
		t.Errorf("expected second message role user, got %s", result.Messages[1].Role)
	}
	if result.MaxTokens == nil || *result.MaxTokens != 1024 {
		t.Errorf("expected max_tokens 1024")
	}
}

func TestAnthropicToOpenAI_NoSystem(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "claude-3",
		MaxTokens: 100,
		Messages: []AnthropicMsg{
			{Role: "user", Content: mustMarshal("Hi")},
		},
	}

	result := AnthropicToOpenAI(req)
	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}
}

func TestOpenAIToAnthropicResponse_TextContent(t *testing.T) {
	stop := "stop"
	resp := &ChatResponse{
		ID:    "chatcmpl-123",
		Model: "qwen3",
		Choices: []ChatChoice{
			{
				Message:      ChatMessage{Content: mustMarshal("Hello back!")},
				FinishReason: &stop,
			},
		},
		Usage: &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}

	result := OpenAIToAnthropicResponse(resp)

	if result.Type != "message" {
		t.Errorf("expected type message, got %s", result.Type)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", result.StopReason)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Text != "Hello back!" {
		t.Errorf("expected text 'Hello back!', got %q", result.Content[0].Text)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected input_tokens 10, got %d", result.Usage.InputTokens)
	}
}

func TestOpenAIToAnthropicResponse_ToolCallsStopReason(t *testing.T) {
	toolCalls := "tool_calls"
	resp := &ChatResponse{
		ID:    "chatcmpl-456",
		Model: "qwen3",
		Choices: []ChatChoice{
			{
				FinishReason: &toolCalls,
			},
		},
	}

	result := OpenAIToAnthropicResponse(resp)
	if result.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %s", result.StopReason)
	}
}

func TestOpenAIToAnthropicResponse_LengthStopReason(t *testing.T) {
	length := "length"
	resp := &ChatResponse{
		Choices: []ChatChoice{{FinishReason: &length}},
	}
	result := OpenAIToAnthropicResponse(resp)
	if result.StopReason != "max_tokens" {
		t.Errorf("expected max_tokens, got %s", result.StopReason)
	}
}

func TestMustMarshal(t *testing.T) {
	result := mustMarshal("hello")
	var s string
	if err := json.Unmarshal(result, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s != "hello" {
		t.Errorf("expected hello, got %s", s)
	}
}
