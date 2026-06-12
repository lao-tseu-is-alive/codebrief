package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

const indexFile = "ai-index.json"

// main parses CLI arguments and dispatches to cmdIndex or cmdQuery.
func main() {
	loadDotEnv(".env")
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	switch os.Args[1] {
	case "index":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: codebrief index <parse_path>")
			os.Exit(1)
		}
		cmdIndex(os.Args[2])
	case "query":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: codebrief query <query_string>")
			os.Exit(1)
		}
		cmdQuery(strings.Join(os.Args[2:], " "))
	default:
		usage()
		os.Exit(1)
	}
}

// loadDotEnv reads KEY=VALUE pairs from path and sets any that are not already
// present in the environment. Shell exports always take precedence over .env.
// Missing or unreadable files are silently ignored.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if key != "" && os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
	_ = scanner.Err()
}

// usage prints available subcommands and their arguments to stderr.
func usage() {
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  codebrief index <parse_path>    # walk and embed a Go codebase, writes ai-index.json")
	fmt.Fprintln(os.Stderr, "  codebrief query <query_string>  # semantic search over ai-index.json")
}

// cmdIndex walks parsePath, prints a Markdown summary of all packages to stdout,
// generates embeddings via OpenAI (if OPENAI_API_KEY is set), and writes ai-index.json.
func cmdIndex(parsePath string) {
	pkgs, err := walkPackages(parsePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Markdown summary to stdout
	fmt.Print("# Go Codebase Index\n\n")
	for _, pkg := range pkgs {
		fmt.Printf("## Package %s (%s)\n", pkg.Name, pkg.Path)
		fmt.Printf("Imports: %v\n", pkg.Imports)
		fmt.Println("\nExported functions:")
		for _, fn := range pkg.Funcs {
			if fn.Exported {
				fmt.Printf("- `%s` (line %d, %s)\n", fn.Signature, fn.Line, fn.File)
				if fn.Doc != "" {
					fmt.Printf("  %s\n", fn.Doc)
				}
			}
		}
		fmt.Println("\nTypes:")
		for _, ti := range pkg.Types {
			fmt.Printf("- %s (%s)\n", ti.Name, ti.Kind)
		}
		fmt.Print("\n")
	}

	chunks := chunksFromIndex(pkgs)
	fmt.Fprintf(os.Stderr, "Building %d chunks...\n", len(chunks))

	embedder, cfg, err := NewEmbedder()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Skipping embeddings: %v\n", err)
	} else {
		if err := embedChunks(chunks, embedder); err != nil {
			fmt.Fprintf(os.Stderr, "Embedding failed: %v\n", err)
		}
	}

	if err := saveIndex(indexFile, pkgs, chunks, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving index: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Saved to %s\n", indexFile)
}

// cmdQuery loads ai-index.json, embeds queryText via OpenAI, and prints the top-5
// chunks ranked by cosine similarity.
func cmdQuery(queryText string) {
	idx, err := loadIndex(indexFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", indexFile, err)
		os.Exit(1)
	}

	embedder, err := newEmbedderFromConfig(idx.Embedder)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	vecs, err := embedder.Embed([]string{queryText})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error embedding query: %v\n", err)
		os.Exit(1)
	}

	results := queryIndex(idx, vecs[0], 5)
	fmt.Printf("Top results for: %q\n\n", queryText)
	for i, c := range results {
		fmt.Printf("--- %d ---\n%s\n\n", i+1, c.Text)
	}
}
