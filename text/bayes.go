/*
Package text holds models which
make text classification easy. They
are regular models, but take strings
as arguments so you can feed in
documents rather than large,
hand-constructed word vectors. Although
models might represent the words as
these vectors, the munging of a
document is hidden from the user.

The simplest model, although suprisingly
effective, is Naive Bayes. If you
want to read more about the specific
model, check out the docs for the
NaiveBayes struct/model.

The following example is an online Naive
Bayes model used for sentiment analysis.

Example Online Naive Bayes Text Classifier (multiclass):

	// create the channel of data and errors
	stream := make(chan base.TextDatapoint, 100)
	errors := make(chan error)

	// make a new NaiveBayes model with
	// 2 classes expected (classes in
	// datapoints will now expect {0,1}.
	// in general, given n as the classes
	// variable, the model will expect
	// datapoint classes in {0,...,n-1})
	//
	// Note that the model is filtering
	// the text to omit anything except
	// words and numbers (and spaces
	// obviously)
	model := NewNaiveBayes(stream, 2, base.OnlyWordsAndNumbers)

	go model.OnlineLearn(errors)

	stream <- base.TextDatapoint{
		X: "I love the city",
		Y: 1,
	}

	stream <- base.TextDatapoint{
		X: "I hate Los Angeles",
		Y: 0,
	}

	stream <- base.TextDatapoint{
		X: "My mother is not a nice lady",
		Y: 0,
	}

	close(stream)

	for {
		err, more := <- errors
		if err != nil {
			fmt.Fprintf(b.Output, "Error passed: %v", err)
		} else {
			// training is done!
			break
		}
	}

	// now you can predict like normal
	class := model.Predict("My mother is in Los Angeles") // 0
*/
package text

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"

	"github.com/Sagleft/goml/base"
	cmap "github.com/orcaman/concurrent-map/v2"
)

/*
NaiveBayes is a general classification
model that calculates the probability
that a datapoint is part of a class
by using Bayes Rule:

	P(y|x) = P(x|y)*P(y)/P(x)

The unique part of this model is that
it assumes words are unrelated to
eachother. For example, the probability
of seeing the word 'penis' in spam emails
if you've already seen 'viagra' might be
different than if you hadn't seen it. The
model ignores this fact because the
computation of full Bayesian model would
take much longer, and would grow significantly
with each word you see.

https://en.wikipedia.org/wiki/Naive_Bayes_classifier
http://cs229.stanford.edu/notes/cs229-notes2.pdf

Based on Bayes Rule, we can easily calculate
the numerator (x | y is just the number of
times x is seen and the class=y, and P(y) is
just the number of times y=class / the number
of positive training examples/words.) The
denominator is also easy to calculate, but
if you recognize that it's just a constant
because it's just the probability of seeing
a certain document given the dataset we can
make the following transformation to be
able to classify without as much classification:

	Class(x) = argmax_c{P(y = c) * ∏P(x|y = c)}

And we can use logarithmic transformations to
make this calculation more computer-practical
(multiplying a bunch of probabilities on [0,1]
will always result in a very small number
which could easily underflow the float value):

	Class(x) = argmax_c{log(P(y = c)) + Σ log(P(x|y = c)0}

Much better. That's our model!
*/
type NaiveBayes struct {
	// Words holds a map of words
	// to their corresponding Word
	// structure
	Words cmap.ConcurrentMap[string, Word]

	// Count holds the number of times
	// class i was seen as Count[i]
	Count []uint64

	// Probabilities holds the probability
	// that class Y is class i as
	// Probabilities[i] for
	Probabilities []float64

	// DocumentCount holds the number of
	// documents that have been seen
	DocumentCount uint64

	// DictCount holds the size of the
	// NaiveBayes model's vocabulary
	DictCount uint64

	// sanitize is used by a model
	// to sanitize input of text
	sanitize transform.Transformer

	// stream holds the datastream
	stream <-chan base.TextDatapoint

	// tokenizer is used by a model
	// to split the input into tokens
	tokenizer Tokenizer

	// Output is the io.Writer used for logging
	// and printing. Defaults to os.Stdout.
	Output io.Writer
}

// To load and store models
type Model struct {
	Words         map[string]Word `json:"words"`
	Count         []uint64        `json:"count"`
	Probabilities []float64       `json:"probabilities"`
	DocumentCount uint64          `json:"document_count"`
	DictCount     uint64          `json:"vocabulary_size"`
}

// Tokenizer accepts a sentence as input and breaks
// it down into a slice of tokens
type Tokenizer interface {
	Tokenize(string) []string
}

// SimpleTokenizer splits sentences
// into tokens delimited by its
// SplitOn string – space, for example
type SimpleTokenizer struct {
	SplitOn string
}

func NewDefaultTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{SplitOn: " "}
}

// Tokenize splits input sentences into a lowecase slice
// of strings. The tokenizer's SlitOn string is used as a
// delimiter and it
func (t *SimpleTokenizer) Tokenize(sentence string) []string {
	// is the tokenizer really the best place to be making
	// everything lowercase? is this really a sanitizaion func?
	return strings.Split(strings.ToLower(sentence), t.SplitOn)
}

// Word holds the structural
// information needed to calculate
// the probability of
type Word struct {
	// Count holds the number of times,
	// (i in Count[i] is the given class)
	Count []uint64

	// Seen holds the number of times
	// the world has been seen. This
	// is than same as
	//    foldl (+) 0 Count
	// in Haskell syntax, but is included
	// you wouldn't have to calculate
	// this every time you wanted to
	// recalc the probabilities (foldl
	// is the same as reduce, basically.)
	Seen uint64

	// DocsSeen is the same as Seen but
	// a word is only counted once even
	// if it's in a document multiple times
	DocsSeen uint64 `json:"-"`
}

type runeSanitizer struct {
	method func(rune) bool
}

func newRuneSanitizer(m func(rune) bool) *runeSanitizer {
	return &runeSanitizer{method: m}
}

func (s *runeSanitizer) Contains(r rune) bool {
	return s.method(r)
}

// NewNaiveBayes returns a NaiveBayes model the
// given number of classes instantiated, ready
// to learn off the given data stream. The sanitization
// function is set to the given function. It must
// comply with the transform.RemoveFunc interface
func NewNaiveBayes(
	stream <-chan base.TextDatapoint,
	classes uint8,
	sanitize func(rune) bool,
) *NaiveBayes {
	return &NaiveBayes{
		Words:         cmap.New[Word](),
		Count:         make([]uint64, classes),
		Probabilities: make([]float64, classes),

		sanitize:  runes.Remove(newRuneSanitizer(sanitize)),
		stream:    stream,
		tokenizer: NewDefaultTokenizer(),

		Output: os.Stdout,
	}
}

// Predict takes in a document, predicts the
// class of the document based on the training
// data passed so far, and returns the class
// estimated for the document.
func (b *NaiveBayes) Predict(sentence string) uint8 {
	sums := make([]float64, len(b.Count))

	sentence, _, _ = transform.String(b.sanitize, sentence)
	words := b.tokenizer.Tokenize(sentence)
	for _, word := range words {
		w, ok := b.Words.Get(word)
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] += math.Log(float64(w.Count[i]+1) / float64(w.Seen+b.DictCount))
		}
	}

	for i := range sums {
		sums[i] += math.Log(b.Probabilities[i])
	}

	// find best class
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}
	}

	return uint8(maxI)
}

// Probability takes in a small document, returns the
// estimated class of the document based on the model
// as well as the probability that the model is part
// of that class
//
// NOTE: you should only use this for small documents
// because, as discussed in the docs for the model, the
// probability will often times underflow because you
// are multiplying together a bunch of probabilities
// which range on [0,1]. As such, the returned float
// could be NaN, and the predicted class could be
// 0 always.
//
// Basically, use Predict to be robust for larger
// documents. Use Probability only on relatively small
// (MAX of maybe a dozen words - basically just
// sentences and words) documents.
func (b *NaiveBayes) Probability(sentence string) (uint8, float64) {
	sums := make([]float64, len(b.Count))
	for i := range sums {
		sums[i] = 1
	}

	sentence, _, _ = transform.String(b.sanitize, sentence)
	words := b.tokenizer.Tokenize(sentence)
	for _, word := range words {
		w, ok := b.Words.Get(word)
		if !ok {
			continue
		}

		for i := range sums {
			sums[i] *= float64(w.Count[i]+1) / float64(w.Seen+b.DictCount)
		}
	}

	for i := range sums {
		sums[i] *= b.Probabilities[i]
	}

	var denom float64
	var maxI int
	for i := range sums {
		if sums[i] > sums[maxI] {
			maxI = i
		}

		denom += sums[i]
	}

	return uint8(maxI), sums[maxI] / denom
}

// OnlineLearn lets the NaiveBayes model learn
// from the datastream, waiting for new data to
// come into the stream from a separate goroutine
func (b *NaiveBayes) OnlineLearn(errors chan<- error) {
	if errors == nil {
		errors = make(chan error)
	}
	if b.stream == nil {
		errors <- fmt.Errorf("ERROR: attempting to learn with nil data stream!\n")
		close(errors)
		return
	}

	fmt.Fprintf(b.Output, "Training:\n\tModel: Multinomial Naïve Bayes\n\tClasses: %v\n", len(b.Count))

	var point base.TextDatapoint
	var more bool

	for {
		point, more = <-b.stream

		if more {
			// sanitize and break up document
			sanitized, _, _ := transform.String(b.sanitize, point.X)
			words := b.tokenizer.Tokenize(sanitized)

			C := int(point.Y)

			if C > len(b.Count)-1 {
				errors <- fmt.Errorf("ERROR: given document class is greater than the number of classes in the model!\n")
				continue
			}

			// update global class probabilities
			b.Count[C]++
			b.DocumentCount++
			for i := range b.Probabilities {
				b.Probabilities[i] = float64(b.Count[i]) / float64(b.DocumentCount)
			}

			// store words seen in document (to add to DocsSeen)
			seenCount := make(map[string]int)

			// update probabilities for words
			for _, word := range words {
				if len(word) < 3 {
					continue
				}

				w, ok := b.Words.Get(word)

				if !ok {
					w = Word{
						Count: make([]uint64, len(b.Count)),
						Seen:  uint64(0),
					}

					b.DictCount++
				}

				w.Count[C]++
				w.Seen++

				b.Words.Set(word, w)

				seenCount[word] = 1
			}

			// add to DocsSeen
			for term := range seenCount {
				tmp, _ := b.Words.Get(term)
				tmp.DocsSeen++
				b.Words.Set(term, tmp)
			}
		} else {
			fmt.Fprintf(b.Output, "Training Completed.\n%v\n\n", b)
			close(errors)
			return
		}
	}
}

// UpdateStream updates the NaiveBayes model's
// text datastream
func (b *NaiveBayes) UpdateStream(stream chan base.TextDatapoint) {
	b.stream = stream
}

// UpdateSanitize updates the NaiveBayes model's
// text sanitization transformation function
func (b *NaiveBayes) UpdateSanitize(sanitize func(rune) bool) {
	b.sanitize = runes.Remove(newRuneSanitizer(sanitize))
}

// UpdateTokenizer updates NaiveBayes model's tokenizer function.
// The default implementation will convert the input to lower
// case and split on the space character.
func (b *NaiveBayes) UpdateTokenizer(tokenizer Tokenizer) {
	b.tokenizer = tokenizer
}

// String implements the fmt interface for clean printing. Here
// we're using it to print the model as the equation h(θ)=...
// where h is the perceptron hypothesis model.
func (b *NaiveBayes) String() string {
	return fmt.Sprintf("h(θ) = argmax_c{log(P(y = c)) + Σlog(P(x|y = c))}\n\tClasses: %v\n\tDocuments evaluated in model: %v\n\tWords evaluated in model: %v\n", len(b.Count), int(b.DocumentCount), int(b.DictCount))
}

func (b *NaiveBayes) ToModel() Model {
	return Model{
		Words:         b.Words.Items(),
		Count:         b.Count,
		Probabilities: b.Probabilities,
		DocumentCount: b.DocumentCount,
		DictCount:     b.DictCount,
	}
}

func NewNaiveBayesFromModel(m Model, sanitize func(rune) bool) *NaiveBayes {
	b := &NaiveBayes{
		Words:         cmap.New[Word](),
		Count:         m.Count,
		Probabilities: m.Probabilities,
		DocumentCount: m.DocumentCount,
		DictCount:     m.DictCount,
		tokenizer:     NewDefaultTokenizer(),
	}

	b.Words.MSet(m.Words)
	b.UpdateSanitize(sanitize)
	return b
}

func NewNaiveBayesFromFile(
	modelFilepath string,
	sanitize func(rune) bool,
	enableGZip bool,
) (*NaiveBayes, error) {
	bytes, err := os.ReadFile(modelFilepath)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}

	if enableGZip {
		bytes, err = decompressJsonBytes(bytes)
		if err != nil {
			return nil, fmt.Errorf("decompress: %w", err)
		}
	}

	var m Model
	if err := json.Unmarshal(bytes, &m); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	return NewNaiveBayesFromModel(m, sanitize), nil
}

func (b *NaiveBayes) PersistModelToFile(
	path string,
	enableGZip bool,
) error {
	if path == "" {
		return errors.New("model path not set")
	}

	dataBytes, err := json.Marshal(b.ToModel())
	if err != nil {
		return fmt.Errorf("encode model: %w", err)
	}

	if enableGZip {
		dataBytes, err = compressJsonBytes(dataBytes)
		if err != nil {
			return fmt.Errorf("compress: %w", err)
		}
	}

	if err := os.WriteFile(path, dataBytes, os.ModePerm); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	return nil
}
