package main

import (
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

// IndexFile is the on-disk format combining the structured package index and embedded chunks.
type IndexFile struct {
	Packages map[string]*PackageInfo `json:"packages"`
	Chunks   []Chunk                 `json:"chunks"`
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

// NewOpenAIEmbedder reads OPENAI_API_KEY from the environment and returns a
// configured embedder using the text-embedding-3-small model.
func NewOpenAIEmbedder() (*OpenAIEmbedder, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}
	return &OpenAIEmbedder{apiKey: key, model: "text-embedding-3-small"}, nil
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
// OLLAMA_MODEL (default nomic-embed-text) from the environment.
func NewOllamaEmbedder() *OllamaEmbedder {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	model := os.Getenv("OLLAMA_MODEL")
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

// NewEmbedder selects an embedder from environment variables.
// EMBEDDER=openai uses OpenAI (requires OPENAI_API_KEY).
// EMBEDDER=ollama uses Ollama (OLLAMA_HOST, OLLAMA_MODEL).
// If EMBEDDER is unset, OpenAI is used when OPENAI_API_KEY is set, otherwise Ollama.
func NewEmbedder() (Embedder, error) {
	provider := os.Getenv("EMBEDDER")
	if provider == "" {
		if os.Getenv("OPENAI_API_KEY") != "" {
			provider = "openai"
		} else {
			provider = "ollama"
		}
	}
	switch provider {
	case "openai":
		return NewOpenAIEmbedder()
	case "ollama":
		return NewOllamaEmbedder(), nil
	default:
		return nil, fmt.Errorf("unknown EMBEDDER %q: use \"openai\" or \"ollama\"", provider)
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

// saveIndex serializes pkgs and chunks into an IndexFile and writes indented JSON to path.
func saveIndex(path string, pkgs map[string]*PackageInfo, chunks []Chunk) error {
	data, err := json.MarshalIndent(IndexFile{Packages: pkgs, Chunks: chunks}, "", "  ")
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
