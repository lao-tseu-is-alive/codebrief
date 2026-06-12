# codebrief

A Go codebase indexer that extracts structural information from Go source files and formats it for use as LLM context. The tool gives an agent a compact, semantically searchable map of a codebase — packages, types with fields, imports, and function signatures with doc comments and file locations — without feeding it raw source.

## Requirements

- Go 1.26+
- An embedding provider (one of):
  - **OpenAI** (default when `OPENAI_API_KEY` is set) — uses `text-embedding-3-small`
  - **Ollama** (default when `OPENAI_API_KEY` is unset) — uses `nomic-embed-text` at `http://localhost:11434`

Without any provider, `index` still writes the structured JSON (vectors will be null) and `query` is unavailable.

### Provider selection

| Env var | Effect |
|---------|--------|
| `LLM_PROVIDER=openai` | Force OpenAI (requires `OPENAI_API_KEY`) |
| `LLM_PROVIDER=ollama` | Force Ollama |
| *(unset)* | OpenAI if `OPENAI_API_KEY` is present, else Ollama |
| `LLM_EMBEDDER_MODEL` | Embedding model name; provider default used if unset (`text-embedding-3-small` / `nomic-embed-text`) |
| `OLLAMA_HOST` | Ollama base URL (default `http://localhost:11434`) |
| `OPENAI_API_KEY` | Required when using OpenAI |

Copy `.env_sample` to `.env` for a starting configuration.

## Usage

```sh
go run . index <parse_path>    # walk a Go codebase, write ai-index.json
go run . query <query_string>  # semantic search over ai-index.json
```

**Examples:**

```sh
# Index this repo
go run . index .

# Index the standard library's net/http package
go run . index $(go env GOROOT)/src/net/http

# Find relevant symbols for a task
go run . query "set a cookie on an HTTP response"
go run . query "how are struct fields extracted from AST"
```

## Output

**stdout** — a Markdown summary for direct use in prompts:
```
# Go Codebase Index

## Package http (src/net/http)
Imports: ["bufio" "context" ...]

Exported functions:
- `Get(url string) (resp *Response, err error)` (line 491, http.go)
  Get issues a GET to the specified URL.

Types:
- Client (struct)
- ResponseWriter (interface)
```

**`ai-index.json`** — a structured file combining the full package index and embedded chunks, suitable for programmatic consumption or retrieval pipelines:

```json
{
  "packages": {
    "./src/net/http": {
      "name": "http",
      "path": "src/net/http",
      "funcs": [
        {
          "name": "Get",
          "signature": "Get(url string) (resp *Response, err error)",
          "doc": "Get issues a GET to the specified URL.",
          "file": "http.go",
          "line": 491,
          "exported": true
        }
      ],
      "types": [
        {
          "name": "Client",
          "kind": "struct",
          "doc": "Client is an HTTP client.",
          "fields": [
            { "name": "Transport", "type": "RoundTripper" },
            { "name": "Timeout", "type": "time.Duration" }
          ]
        },
        {
          "name": "ResponseWriter",
          "kind": "interface",
          "methods": ["Header", "Write", "WriteHeader"]
        }
      ],
      "imports": ["\"bufio\"", "\"context\""]
    }
  },
  "chunks": [
    {
      "id": "pkg:src/net/http/func:Get",
      "text": "Package: http (src/net/http)\nFunc: Get(url string) (resp *Response, err error)\nDoc: Get issues a GET...\nLocation: http.go:491",
      "vector": [0.021, -0.004, ...],
      "metadata": { "package": "http", "kind": "func", "file": "http.go", "line": "491" }
    }
  ]
}
```

If `OPENAI_API_KEY` is not set, `vector` fields are `null` and `query` is unavailable, but the structured index is still fully written.

## How it works

**Indexing pipeline:**

1. `go/parser` builds an AST for each `.go` file (skips `vendor/`, no compilation needed)
2. Per file: extracts function declarations (signature via `go/printer`, doc comment, file + line), type declarations (struct fields, interface methods, kind), and imports
3. Aggregates per directory path into `PackageInfo` structs; imports are deduplicated across files
4. Flattens into `[]Chunk` — one chunk per exported function or type — formatted as human-readable text
5. Sends chunks to OpenAI `text-embedding-3-small` in batches of 100
6. Writes `ai-index.json` with both the structured packages and the embedded chunks

**Query pipeline:**

1. Loads `ai-index.json`
2. Embeds the query string using the same model
3. Ranks all chunks by cosine similarity (in-memory)
4. Prints the top-5 matching chunks with their text

## Architecture

```
codebrief/
  main.go        CLI entry point — index and query subcommands
  indexer.go     AST walking, data types (PackageInfo, Func, TypeInfo, FieldInfo)
  retrieval.go   Chunk, Embedder interface, OpenAIEmbedder, OllamaEmbedder, NewEmbedder factory, vector store, cosine similarity
```

The `Embedder` interface makes it straightforward to swap in a different provider:

```go
type Embedder interface {
    Embed(texts []string) ([][]float32, error)
}
```

## Limitations

- Packages are keyed by directory path, so the output map key is a filesystem path, not a Go import path
- Parse errors are silently skipped with a stderr warning; malformed files produce no output
- The in-memory vector store loads all embeddings on each query — not suitable for very large corpora
