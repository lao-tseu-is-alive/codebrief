package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type PackageInfo struct {
	Name    string     `json:"name"`
	Path    string     `json:"path"`
	Funcs   []Func     `json:"funcs"`
	Types   []TypeInfo `json:"types"`
	Imports []string   `json:"imports"`
}

type Func struct {
	Name      string `json:"name"`
	Signature string `json:"signature"`
	Doc       string `json:"doc,omitempty"`
	File      string `json:"file"`
	Line      int    `json:"line"`
	Exported  bool   `json:"exported"`
}

type TypeInfo struct {
	Name    string      `json:"name"`
	Kind    string      `json:"kind"` // struct, interface, alias, other
	Doc     string      `json:"doc,omitempty"`
	Fields  []FieldInfo `json:"fields,omitempty"`
	Methods []string    `json:"methods,omitempty"`
}

type FieldInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// walkPackages recursively walks parsePath, parses every non-vendor .go file,
// and returns a map keyed by directory path containing each package's functions,
// types, and deduplicated imports.
func walkPackages(parsePath string) (map[string]*PackageInfo, error) {
	fset := token.NewFileSet()
	pkgs := make(map[string]*PackageInfo)
	importSets := make(map[string]map[string]struct{})

	err := filepath.Walk(parsePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor/") {
			return nil
		}

		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", path, err)
			return nil
		}

		pkgPath := filepath.Dir(path)
		if _, exists := pkgs[pkgPath]; !exists {
			pkgs[pkgPath] = &PackageInfo{Name: f.Name.Name, Path: pkgPath}
			importSets[pkgPath] = make(map[string]struct{})
		}

		for _, decl := range f.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				startLine := fset.Position(fn.Pos()).Line

				recv := ""
				if fn.Recv != nil {
					recv = "(" + recvString(fset, fn.Recv) + ") "
				}
				params := fieldListString(fset, fn.Type.Params)
				if params == "" {
					params = "()"
				}
				results := fieldListString(fset, fn.Type.Results)
				sig := recv + fn.Name.Name + params
				if results != "" {
					sig += " " + results
				}

				doc := ""
				if fn.Doc != nil {
					doc = strings.TrimSpace(fn.Doc.Text())
				}

				pkgs[pkgPath].Funcs = append(pkgs[pkgPath].Funcs, Func{
					Name:      fn.Name.Name,
					Signature: sig,
					Doc:       doc,
					File:      path,
					Line:      startLine,
					Exported:  fn.Name.IsExported(),
				})
			}

			if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.TYPE {
				for _, spec := range genDecl.Specs {
					if ts, ok := spec.(*ast.TypeSpec); ok {
						ti := TypeInfo{Name: ts.Name.Name}

						if ts.Doc != nil {
							ti.Doc = strings.TrimSpace(ts.Doc.Text())
						} else if genDecl.Doc != nil {
							ti.Doc = strings.TrimSpace(genDecl.Doc.Text())
						}

						switch t := ts.Type.(type) {
						case *ast.StructType:
							ti.Kind = "struct"
							for _, field := range t.Fields.List {
								typStr := exprString(fset, field.Type)
								if len(field.Names) == 0 {
									ti.Fields = append(ti.Fields, FieldInfo{Name: typStr, Type: typStr})
								} else {
									for _, n := range field.Names {
										ti.Fields = append(ti.Fields, FieldInfo{Name: n.Name, Type: typStr})
									}
								}
							}
						case *ast.InterfaceType:
							ti.Kind = "interface"
							for _, method := range t.Methods.List {
								for _, n := range method.Names {
									ti.Methods = append(ti.Methods, n.Name)
								}
							}
						case *ast.Ident, *ast.SelectorExpr, *ast.ArrayType, *ast.MapType:
							ti.Kind = "alias"
						default:
							ti.Kind = "other"
						}

						pkgs[pkgPath].Types = append(pkgs[pkgPath].Types, ti)
					}
				}
			}
		}

		for _, imp := range f.Imports {
			importSets[pkgPath][imp.Path.Value] = struct{}{}
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	for pkgPath, pkg := range pkgs {
		for imp := range importSets[pkgPath] {
			pkg.Imports = append(pkg.Imports, imp)
		}
		sort.Strings(pkg.Imports)
	}

	return pkgs, nil
}

// exprString uses go/printer to render an AST expression node back to source text.
func exprString(fset *token.FileSet, expr ast.Expr) string {
	if expr == nil {
		return ""
	}
	var buf bytes.Buffer
	printer.Fprint(&buf, fset, expr)
	return buf.String()
}

// fieldListString renders an *ast.FieldList as a parenthesized, comma-separated
// parameter or result string (e.g. "(ctx context.Context, n int)").
func fieldListString(fset *token.FileSet, fl *ast.FieldList) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	var parts []string
	for _, field := range fl.List {
		typStr := exprString(fset, field.Type)
		if len(field.Names) == 0 {
			parts = append(parts, typStr)
		} else {
			names := make([]string, len(field.Names))
			for i, n := range field.Names {
				names[i] = n.Name
			}
			parts = append(parts, strings.Join(names, ", ")+" "+typStr)
		}
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// recvString returns the type string for the first element of a method receiver
// list (e.g. "*OpenAIEmbedder"), used when building a function signature.
func recvString(fset *token.FileSet, recv *ast.FieldList) string {
	if recv == nil || len(recv.List) == 0 {
		return ""
	}
	return exprString(fset, recv.List[0].Type)
}
