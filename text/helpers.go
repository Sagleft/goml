package text

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
)

func compressJsonBytes(jsonBytes []byte) ([]byte, error) {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)

	if _, err := gz.Write(jsonBytes); err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}
	if err := gz.Close(); err != nil {
		return nil, fmt.Errorf("close: %w", err)
	}

	return buf.Bytes(), nil
}

func decompressJsonBytes(gzBytes []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(gzBytes))
	if err != nil {
		return nil, fmt.Errorf("create reader: %w", err)
	}
	defer r.Close()

	decompressed, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read all: %w", err)
	}
	return decompressed, nil
}
