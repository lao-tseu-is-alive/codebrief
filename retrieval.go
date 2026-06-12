package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
)

// Chunk is a single embeddable unit from the index — one function or one type.
type Chunk struct {
	ID       string            `json:"id"`
	Text     string            `json:"text"`
	Vector   []float32         `json:"vector"`
	Metadata map[string]string `json:"metadata"`
}

// EmbedderConfig records which provider and model produced the vectors in an IndexFile.
type EmbedderConfig struct {
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
}

// IndexFile is the on-disk format combining the structured package index and embedded chunks.
type IndexFile struct {
	Packages map[string]*PackageInfo `json:"packages"`
	Chunks   []Chunk                 `json:"chunks"`
	Embedder EmbedderConfig          `json:"embedder"`
}

// Embedder is the interface for converting text to embedding vectors.
// Swap implementations to change providers.
type Embedder interface {
	Embed(texts []string) ([][]float32, error)
}

// OpenAIEmbedder calls the OpenAI embeddings API using net/http (no external deps).
type OpenAIEmbedder struct {
	apiKey string
	model  string
}

// NewOpenAIEmbedder reads OPENAI_API_KEY and LLM_EMBEDDER_MODEL (default text-embedding-3-small)
// from the environment and returns a configured embedder.
func NewOpenAIEmbedder() (*OpenAIEmbedder, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}
	model := os.Getenv("LLM_EMBEDDER_MODEL")
	if model == "" {
		model = "text-embedding-3-small"
	}
	return &OpenAIEmbedder{apiKey: key, model: model}, nil
}

// Embed sends texts to the OpenAI embeddings API and returns one float32 vector
// per input in the same order, using net/http with no external dependencies.
func (e *OpenAIEmbedder) Embed(texts []string) ([][]float32, error) {
	type request struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	type embeddingItem struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	}
	type response struct {
		Data []embeddingItem `json:"data"`
	}

	body, _ := json.Marshal(request{Model: e.model, Input: texts})
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody map[string]any
		json.NewDecoder(resp.Body).Decode(&errBody)
		return nil, fmt.Errorf("OpenAI API error %d: %v", resp.StatusCode, errBody)
	}

	var result response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	sort.Slice(result.Data, func(i, j int) bool {
		return result.Data[i].Index < result.Data[j].Index
	})

	vectors := make([][]float32, len(result.Data))
	for i, item := range result.Data {
		vectors[i] = item.Embedding
	}
	return vectors, nil
}

// OllamaEmbedder calls the Ollama /api/embed endpoint (no external deps).
type OllamaEmbedder struct {
	host  string
	model string
}

// NewOllamaEmbedder reads OLLAMA_HOST (default http://localhost:11434) and
// LLM_EMBEDDER_MODEL (default nomic-embed-text) from the environment.
func NewOllamaEmbedder() *OllamaEmbedder {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	model := os.Getenv("LLM_EMBEDDER_MODEL")
	if model == "" {
		model = "nomic-embed-text"
	}
	return &OllamaEmbedder{host: host, model: model}
}

// Embed sends texts to the Ollama /api/embed endpoint and returns one float32
// vector per input in the same order.
func (e *OllamaEmbedder) Embed(texts []string) ([][]float32, error) {
	type request struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	type response struct {
		Embeddings [][]float32 `json:"embeddings"`
	}

	body, _ := json.Marshal(request{Model: e.model, Input: texts})
	req, err := http.NewRequest("POST", e.host+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Ollama unreachable at %s: %w", e.host, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody map[string]any
		json.NewDecoder(resp.Body).Decode(&errBody)
		return nil, fmt.Errorf("Ollama API error %d: %v", resp.StatusCode, errBody)
	}

	var result response
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Embeddings, nil
}

// NewEmbedder selects an embedder from environment variables and returns it alongside
// its config so the caller can persist provider/model in the index.
// LLM_PROVIDER=openai uses OpenAI (requires OPENAI_API_KEY).
// LLM_PROVIDER=ollama uses Ollama (OLLAMA_HOST, LLM_EMBEDDER_MODEL).
// If LLM_PROVIDER is unset, OpenAI is used when OPENAI_API_KEY is set, otherwise Ollama.
func NewEmbedder() (Embedder, EmbedderConfig, error) {
	provider := os.Getenv("LLM_PROVIDER")
	if provider == "" {
		if os.Getenv("OPENAI_API_KEY") != "" {
			provider = "openai"
		} else {
			provider = "ollama"
		}
	}
	switch provider {
	case "openai":
		e, err := NewOpenAIEmbedder()
		if err != nil {
			return nil, EmbedderConfig{}, err
		}
		return e, EmbedderConfig{Provider: "openai", Model: e.model}, nil
	case "ollama":
		e := NewOllamaEmbedder()
		return e, EmbedderConfig{Provider: "ollama", Model: e.model}, nil
	default:
		return nil, EmbedderConfig{}, fmt.Errorf("unknown LLM_PROVIDER %q: use \"openai\" or \"ollama\"", provider)
	}
}

// newEmbedderFromConfig recreates an embedder from the config stored in an IndexFile,
// ensuring query uses the same provider and model that built the index.
// Falls back to NewEmbedder() for index files written before this field existed.
func newEmbedderFromConfig(cfg EmbedderConfig) (Embedder, error) {
	switch cfg.Provider {
	case "openai":
		return NewOpenAIEmbedder()
	case "ollama":
		e := NewOllamaEmbedder()
		if cfg.Model != "" {
			e.model = cfg.Model
		}
		return e, nil
	default:
		e, _, err := NewEmbedder()
		return e, err
	}
}

// chunksFromIndex flattens the package index into one Chunk per exported function or type.
func chunksFromIndex(pkgs map[string]*PackageInfo) []Chunk {
	var chunks []Chunk
	for _, pkg := range pkgs {
		for _, fn := range pkg.Funcs {
			if !fn.Exported {
				continue
			}
			var sb strings.Builder
			fmt.Fprintf(&sb, "Package: %s (%s)\n", pkg.Name, pkg.Path)
			fmt.Fprintf(&sb, "Func: %s\n", fn.Signature)
			if fn.Doc != "" {
				fmt.Fprintf(&sb, "Doc: %s\n", fn.Doc)
			}
			fmt.Fprintf(&sb, "Location: %s:%d", fn.File, fn.Line)

			chunks = append(chunks, Chunk{
				ID:   fmt.Sprintf("pkg:%s/func:%s", pkg.Path, fn.Name),
				Text: sb.String(),
				Metadata: map[string]string{
					"package": pkg.Name,
					"kind":    "func",
					"file":    fn.File,
					"line":    fmt.Sprintf("%d", fn.Line),
				},
			})
		}
		for _, ti := range pkg.Types {
			var sb strings.Builder
			fmt.Fprintf(&sb, "Package: %s (%s)\n", pkg.Name, pkg.Path)
			fmt.Fprintf(&sb, "Type: %s (%s)\n", ti.Name, ti.Kind)
			if ti.Doc != "" {
				fmt.Fprintf(&sb, "Doc: %s\n", ti.Doc)
			}
			if len(ti.Fields) > 0 {
				fieldParts := make([]string, len(ti.Fields))
				for i, f := range ti.Fields {
					fieldParts[i] = f.Name + " " + f.Type
				}
				fmt.Fprintf(&sb, "Fields: %s\n", strings.Join(fieldParts, ", "))
			}
			if len(ti.Methods) > 0 {
				fmt.Fprintf(&sb, "Methods: %s\n", strings.Join(ti.Methods, ", "))
			}

			chunks = append(chunks, Chunk{
				ID:   fmt.Sprintf("pkg:%s/type:%s", pkg.Path, ti.Name),
				Text: sb.String(),
				Metadata: map[string]string{
					"package": pkg.Name,
					"kind":    "type:" + ti.Kind,
				},
			})
		}
	}
	return chunks
}

// embedChunks fills Vector on each chunk by calling the embedder in batches.
func embedChunks(chunks []Chunk, embedder Embedder) error {
	const batchSize = 100
	for i := 0; i < len(chunks); i += batchSize {
		end := min(i+batchSize, len(chunks))
		batch := chunks[i:end]
		texts := make([]string, len(batch))
		for j, c := range batch {
			texts[j] = c.Text
		}
		vectors, err := embedder.Embed(texts)
		if err != nil {
			return fmt.Errorf("embedding batch %d-%d: %w", i, end, err)
		}
		for j := range batch {
			chunks[i+j].Vector = vectors[j]
		}
		fmt.Fprintf(os.Stderr, "embedded %d/%d chunks\n", end, len(chunks))
	}
	return nil
}

// saveIndex serializes pkgs, chunks, and the embedder config into an IndexFile and writes indented JSON to path.
func saveIndex(path string, pkgs map[string]*PackageInfo, chunks []Chunk, cfg EmbedderConfig) error {
	data, err := json.MarshalIndent(IndexFile{Packages: pkgs, Chunks: chunks, Embedder: cfg}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// loadIndex reads and deserializes an IndexFile from the JSON file at path.
func loadIndex(path string) (*IndexFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f IndexFile
	return &f, json.Unmarshal(data, &f)
}

// queryIndex returns the top-K chunks by cosine similarity to queryVec.
func queryIndex(idx *IndexFile, queryVec []float32, topK int) []Chunk {
	type scored struct {
		chunk Chunk
		score float32
	}
	scores := make([]scored, 0, len(idx.Chunks))
	for _, c := range idx.Chunks {
		if len(c.Vector) > 0 {
			scores = append(scores, scored{c, cosine(queryVec, c.Vector)})
		}
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	result := make([]Chunk, 0, topK)
	for i := 0; i < topK && i < len(scores); i++ {
		result = append(result, scores[i].chunk)
	}
	return result
}

// cosine returns the cosine similarity between two equal-length float32 vectors.
// Returns 0 if either vector has zero magnitude.
func cosine(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

// generateAnswer sends the retrieved chunks as context to LLM_QUERY_MODEL via the
// Ollama /api/chat endpoint and streams the generated answer to stdout.
func generateAnswer(query string, chunks []Chunk, model, host string) error {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type request struct {
		Model    string    `json:"model"`
		Messages []message `json:"messages"`
		Stream   bool      `json:"stream"`
	}
	type responseChunk struct {
		Message message `json:"message"`
		Done    bool    `json:"done"`
	}

	var ctx strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&ctx, "--- %d ---\n%s\n\n", i+1, c.Text)
	}

	body, _ := json.Marshal(request{
		Model: model,
		Messages: []message{
			{Role: "system", Content: "You are a code assistant. Answer the user's question using only the following code context.\n\n" + ctx.String()},
			{Role: "user", Content: query},
		},
		Stream: true,
	})

	req, err := http.NewRequest("POST", host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("Ollama unreachable at %s: %w", host, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody map[string]any
		json.NewDecoder(resp.Body).Decode(&errBody)
		return fmt.Errorf("Ollama API error %d: %v", resp.StatusCode, errBody)
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var chunk responseChunk
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		fmt.Print(chunk.Message.Content)
		if chunk.Done {
			fmt.Println()
			break
		}
	}
	return scanner.Err()
}
