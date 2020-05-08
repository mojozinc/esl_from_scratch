package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

type datarecord struct {
	vector []interface{}
}

type dataframe struct {
	columns    []string
	column2idx map[string]int
	records    []datarecord
}

func NewDataFrame(columns []string) dataframe {
	//dataframe constructor
	var df dataframe
	df.columns = columns
	df.column2idx = make(map[string]int)
	for i, col := range df.columns {
		df.column2idx[col] = i
	}
	return df
}

func (df *dataframe) addRow(vector []interface{}) {
	var dr datarecord
	dr.vector = vector
	df.records = append(df.records, dr)
}

type featureEncoder struct {
	encoder map[string]map[string]int
	max     map[string]int
	decoder map[string][]string
}

func (rec *datarecord) vectorize(encoder *featureEncoder, columns []string) (Vector []float64) {
	for i, column := range columns {
		encoderTmp, ok := encoder.encoder[column]
		if ok {
			Vector = append(Vector, float64(encoderTmp[rec.vector[i].(string)]+1))
		} else {
			Vector = append(Vector, rec.vector[i].(float64))
		}
	}
	return
}

func buildEncoder(df dataframe) (encoder featureEncoder) {
	buildencoder := func(values []string) (int, map[string]int, []string) {
		featureCount := 0
		featureMap := make(map[string]int)
		var decoder []string
		for _, feature := range values {
			_, ok := featureMap[feature]
			if !ok {
				decoder = append(decoder, feature)
				featureMap[feature] = featureCount
				featureCount++
			}
		}
		return featureCount, featureMap, decoder
	}

	fieldValues := make(map[string][]string)
	encoder.encoder = make(map[string]map[string]int)
	encoder.max = make(map[string]int)
	encoder.decoder = make(map[string][]string)
	for i, col := range df.columns {
		for _, rec := range df.records {
			val, ok := rec.vector[i].(string)
			if ok {
				fieldValues[col] = append(fieldValues[col], val)
			}
		}
		if len(fieldValues[col]) > 0 {
			encoder.max[col], encoder.encoder[col], encoder.decoder[col] = buildencoder(fieldValues[col])
		}
	}
	return
}

func testTrainSplit(data dataframe, split float32) (testData dataframe, trainData dataframe) {
	rand.Seed(time.Now().Unix())
	rand.Shuffle(len(data.records), func(i, j int) { data.records[i], data.records[j] = data.records[j], data.records[i] })
	NTest := int(split * float32(len(data.records)))
	testData = NewDataFrame(data.columns)
	trainData = NewDataFrame(data.columns)
	testData.records = data.records[:NTest]
	trainData.records = data.records[NTest:]
	return
}

func (df *dataframe) vectorize(encoder *featureEncoder, outputColumn string, model string) (*mat.Dense, *mat.Dense) {
	outputColumnInt := df.column2idx[outputColumn]
	N, p := len(df.records), len(df.columns)-1
	X := mat.NewDense(N, p, nil)
	var yTmp []float64
	for i, datapoint := range df.records {
		vector := datapoint.vectorize(encoder, df.columns)
		X.SetRow(i, append(vector[:outputColumnInt], vector[outputColumnInt+1:]...))
		yTmp = append(yTmp, vector[outputColumnInt])
	}
	var outputEncoder func(float64) *mat.Dense
	var Y *mat.Dense
	if model == "classification" {
		var classes int = encoder.max[outputColumn]
		outputEncoder = func(val float64) *mat.Dense {
			vec := mat.NewDense(1, classes, nil)
			for i := 0; i < classes; i++ {
				if float64(i) == val {
					vec.Set(0, i, 1.0)
				} else {
					vec.Set(0, i, 0.0)
				}
			}
			return vec
		}
		Y = mat.NewDense(N, classes, nil)
	} else if model == "regression" {
		outputEncoder = func(val float64) *mat.Dense { return mat.NewDense(1, 1, []float64{val}) }
		Y = mat.NewDense(N, 1, nil)
	}
	for i := 0; i < N; i++ {
		Y.SetRow(i, outputEncoder(yTmp[i]).RawRowView(0))
	}
	return X, Y
}

func residualSquareSum(Y, yPredict *mat.Dense) float64 {
	var deltaMat mat.Dense
	deltaMat.Sub(yPredict, Y)
	deltaMat.MulElem(&deltaMat, &deltaMat)
	return mat.Sum(&deltaMat)
}

type linearModel struct {
	modelType   string
	outputLabel string
	weights     *mat.Dense
}

func (model *linearModel) learn(df dataframe) *mat.Dense {
	onevec := func(n int) []float64 {
		var vec []float64
		for i := 0; i < n; i++ {
			vec = append(vec, 1.0)
		}
		return vec
	}
	fit := func(X, Y *mat.Dense) mat.Dense {
		// minimise the expression (y - b.x)^2
		// solve for beta = (x^t.x)^-1.x^t.y
		var x0, x1, x2, x3, Beta mat.Dense

		// add a bias column
		r, _ := X.Dims()
		x0.Augment(X, mat.NewDense(r, 1, onevec(r)))
		x1.Mul(x0.T(), &x0)
		x2.Inverse(&x1)
		x3.Mul(&x2, x0.T())
		Beta.Mul(&x3, Y)
		return Beta
	}

	predict := func(X *mat.Dense, weights *mat.Dense) mat.Dense {
		var predictors *mat.Dense
		var p mat.Dense
		// add a bias column
		r, _ := X.Dims()
		p.Augment(X, mat.NewDense(r, 1, onevec(r)))
		predictors = &p
		var yPredict mat.Dense
		yPredict.Mul(predictors, weights)
		activationFunction := func(vec []float64) []float64 {
			max, i := vec[0], 0
			for j, x := range vec {
				if x > max {
					max, i = x, j
				}
			}
			var activatedVec []float64
			for j := range vec {
				if j == i {
					activatedVec = append(activatedVec, 1.0)
				} else {
					activatedVec = append(activatedVec, 0.0)
				}
			}
			return activatedVec
		}
		r, _ = yPredict.Dims()
		if model.modelType == "classification" {
			for i := 0; i < r; i++ {
				yPredict.SetRow(i, activationFunction(yPredict.RawRowView(i)))
			}
		}
		return yPredict
	}

	encoder := buildEncoder(df)
	decodeClassLabel := func(onehot []float64) (label string) {
		activeClass := -1
		for i, x := range onehot {
			if x == 1.0 {
				activeClass = i
				break
			}
		}
		label = encoder.decoder[model.outputLabel][activeClass]
		return label
	}
	testData, trainData := testTrainSplit(df, 0.3)
	if model.modelType == "classification" && encoder.max[model.outputLabel] < 2 {
		log.Printf("Nothing to fit, need atleast 2 classes in training data")
		return nil
	}
	testX, testY := testData.vectorize(&encoder, model.outputLabel, model.modelType)
	trainX, trainY := trainData.vectorize(&encoder, model.outputLabel, model.modelType)
	// if model.modelType == "regression" {
	// 	testY.Apply(func(i, j int, x float64) float64 { return 5 * x }, testY)
	// 	trainY.Apply(func(i, j int, x float64) float64 { return 5 * x }, trainY)
	// }
	weights := fit(trainX, trainY)
	model.weights = &weights
	yPredict := predict(testX, &weights)
	r, _ := testY.Dims()
	failures := 0
	for i := 0; i < r && failures < 1000; i++ {
		switch model.modelType {
		case "classification":
			tx, px := decodeClassLabel(testY.RawRowView(i)), decodeClassLabel(yPredict.RawRowView(i))
			if tx != px {
				// fmt.Printf("predictors: %v, class: %s, predicted: %s\n", testX.RawRowView(i), tx, px)
				failures++
			}
		case "regression":
			tx, px := testY.RawRowView(i)[0], yPredict.RawRowView(i)[0]
			if math.Pow(tx-px, 2) > math.Pow(0.1, 2) {
				// fmt.Printf("predictors: %v, expected: %v, predicted: %v\n", testX.RawRowView(i), tx, px)
				failures++
			}
		}
	}
	return model.weights
}

func loadDataFromCSV(datasetPath string, columns []columnDesc) (df dataframe) {
	fp, err := os.Open(datasetPath)
	if err != nil {
		panic(err)
	}
	defer fp.Close()
	r := csv.NewReader(fp)
	r.Read()
	var columnNames []string
	var columnIdxes []int
	for _, pair := range columns {
		columnNames = append(columnNames, pair.str)
		columnIdxes = append(columnIdxes, pair.i)
	}
	df = NewDataFrame(columnNames)
	var row []interface{}
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		row = make([]interface{}, len(columns))
		for i, colDesc := range columns {
			if colDesc.parser != nil {
				row[i] = colDesc.parser(record[colDesc.i])
			} else {
				row[i] = record[colDesc.i]
			}
		}
		df.addRow(row)
	}
	return
}

type columnDesc struct {
	str    string
	i      int
	parser func(value string) interface{}
}

func main() {
	headerDesc := []columnDesc{
		{"species", 1, nil},
		{"part", 2, nil},
		{"measure", 3, nil},
		{"value", 4, func(value string) interface{} {
			val, err := strconv.ParseFloat(value, 64)
			if err != nil {
				log.Fatal(err)
			}
			return val
		},
		},
	}
	data := loadDataFromCSV("iris_tidy.csv", headerDesc)
	var model linearModel
	var weights *mat.Dense
	model = linearModel{"regression", "species", nil}
	weights = model.learn(data)
	fmt.Printf("learned these weights for regression model\n%v\n\n", mat.Formatted(weights))

	model = linearModel{"classification", "species", nil}
	weights = model.learn(data)
	if weights != nil {
		fmt.Printf("learned these weights for classification model\n%v\n\n", mat.Formatted(weights))
	}
}
