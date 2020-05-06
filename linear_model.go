package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

type CSVRecord struct {
	//Represent a csvRecord
	species string // 1 for setosa,
	part    string // 1.0 for Sepal, 2.0 for petal
	measure string // 1.0 for length, 2.0 for width
	value   float64
}

type datarecord struct {
	columns    *[]string
	vector     []interface{}
	column2idx *map[string]int
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
	dr.columns = &df.columns
	dr.column2idx = &df.column2idx
	df.records = append(df.records, dr)
}

type featureEncoder struct {
	encoder map[string]map[string]int
	max     map[string]int
}

func (rec *datarecord) vectorize(encoder *featureEncoder) (Vector []float64) {
	for i, column := range *rec.columns {
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
	buildencoder := func(values []string) (int, map[string]int) {
		featureCount := 0
		featureMap := make(map[string]int)
		for _, feature := range values {
			_, ok := featureMap[feature]
			if !ok {
				featureMap[feature] = featureCount
				featureCount++
			}
		}
		return featureCount, featureMap
	}

	fieldValues := make(map[string][]string)
	encoder.encoder = make(map[string]map[string]int)
	encoder.max = make(map[string]int)
	for i, col := range df.columns {
		for _, rec := range df.records {
			val, ok := rec.vector[i].(string)
			if ok {
				fieldValues[col] = append(fieldValues[col], val)
			}
		}
		if len(fieldValues[col]) > 0 {
			encoder.max[col], encoder.encoder[col] = buildencoder(fieldValues[col])
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
		vector := datapoint.vectorize(encoder)
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
	var deltaMat, rss mat.Dense
	deltaMat.Sub(yPredict, Y)
	rss.Mul(deltaMat.T(), &deltaMat)
	return rss.At(0, 0)
}

func predict(predictors *mat.Dense, weights *mat.Dense) mat.Dense {
	var Ypredict mat.Dense
	Ypredict.Mul(predictors, weights)
	return Ypredict
}

func learn(X, Y *mat.Dense) mat.Dense {
	// solve for beta = (x^t.x)^-1.x^t.y
	var x0, x1, Beta mat.Dense
	x0.Mul(X.T(), X)
	x0.Inverse(&x0)
	x1.Mul(&x0, X.T())
	Beta.Mul(&x1, Y)
	return Beta
}
func linearModel(df dataframe) {
	encoder := buildEncoder(df)
	testData, trainData := testTrainSplit(df, 0.3)
	testX, testY := testData.vectorize(&encoder, "species", "classification")
	trainX, trainY := trainData.vectorize(&encoder, "species", "classification")

	weights := learn(trainX, trainY)
	yPredict := predict(testX, &weights)
	rss := residualSquareSum(testY, &yPredict)
	var yaugmented mat.Dense
	yaugmented.Augment(testY, &yPredict)
	fmt.Println("residual square sum", rss)
	fmt.Printf("Prediction\n%v\n", mat.Formatted(&yaugmented))
}

func loadData(datasetPath string) (df dataframe) {
	fp, err := os.Open(datasetPath)
	if err != nil {
		panic(err)
	}
	defer fp.Close()
	r := csv.NewReader(fp)
	r.Read()
	df = NewDataFrame([]string{"species", "part", "measure", "value"})
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		val, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			log.Fatal(err)
		}
		df.addRow([]interface{}{record[1], record[2], record[3], val})
	}
	return
}

func main() {
	data := loadData("iris_tidy.csv")
	linearModel(data)
}
