# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Go CLI tool that indexes a Go codebase via AST parsing and produces a semantically searchable index for LLM context. It uses `go/ast`/`go/parser`/`go/printer` (stdlib only) to extract packages, types with fields, exported function signatures with doc comments, and imports. Embeddings are generated via the OpenAI API (`text-embedding-3-small`) and stored alongside the structured index for cosine-similarity retrieval.

## Build & run

```sh
go run . index <parse_path>    # e.g. go run . index .
go run . query <query_string>  # requires ai-index.json from a prior index run
go build -o codebrief .
```

`OPENAI_API_KEY` must be set for embeddings and query. Without it, `index` still writes the structured JSON (vectors will be null).

## File layout

| File | Responsibility |
|------|---------------|
| `main.go` | CLI dispatch — `index` and `query` subcommands |
| `indexer.go` | `walkPackages()` — AST walking + all data types (`PackageInfo`, `Func`, `TypeInfo`, `FieldInfo`) + AST printer helpers |
| `retrieval.go` | `Chunk`, `Embedder` interface, `OpenAIEmbedder`, `chunksFromIndex()`, `embedChunks()`, `saveIndex`/`loadIndex`, `queryIndex`, `cosine()` |

## Key data flow

**`index` subcommand:**
1. `walkPackages(parsePath)` → `map[string]*PackageInfo` keyed by directory path
2. `chunksFromIndex(pkgs)` → `[]Chunk` (one per exported func or type)
3. `embedChunks(chunks, embedder)` → fills `Chunk.Vector` via OpenAI, batched in 100s
4. `saveIndex("ai-index.json", pkgs, chunks)` → writes `IndexFile{Packages, Chunks}`

**`query` subcommand:**
1. `loadIndex("ai-index.json")` → `*IndexFile`
2. `embedder.Embed([]string{query})` → single vector
3. `queryIndex(idx, vec, 5)` → cosine similarity rank, top-5 chunks printed

## Extending

To add a new embedding provider, implement the `Embedder` interface in `retrieval.go`:
```go
type Embedder interface {
    Embed(texts []string) ([][]float32, error)
}
```

To add package-level `const`/`var` extraction, extend the `genDecl` branch in `walkPackages()` in `indexer.go` for `token.CONST` and `token.VAR`.

## Known limitations

- Package map is keyed by filesystem directory path, not Go import path — two separate module roots can collide if their relative paths match
- Parse errors are skipped silently (warning to stderr); malformed files produce no entries
- In-memory vector store loads all embeddings per query — not suitable for very large corpora without chunking the store
