package main

import (
	"fmt" 
	"sync"
	"time"
	// "runtime"
	"math/rand"
	"C"
)

//export get_data_matrix_list
func get_data_matrix_list(data_matrx [][]int8)[]int8 {
	dc := make([]int8, len(data_matrx))
	for j:= range((data_matrx[0])){
		for k := range((data_matrx)){
			dc[k] = data_matrx[k][j]
		}
	}
	return dc
}
//export multiply_matrix
func multiply_matrix (weight_matrix [][]int8,data_matrix [][]int8 ) [][]int16 {
	// var wg sync.WaitGroup
	// var div uint32 = v/uint32(runtime.NumCPU())
	// var k,i uint32;
	// for k:=0; k < len(data) ; k++ {
	// 	for i=0; i<v-div; i=i+div {
	// 		//fmt.Printf("value range of i %d to %d \n", i,i+div)
	// 		wg.Add(1);
	// 		go internal_loop(dists, v, k,i,i+div,&wg)
	// 	}
	// 	wg.Add(1)
	// 	go internal_loop(dists, v, k,i,v,&wg)
	// 	wg.Wait()
	// }
	result := makeresultmatrix(len(weight_matrix), len(data_matrix[0]))
	dc :=get_data_matrix_list(data_matrix)
	for i := range(weight_matrix){
		temp_zip := int16(0)
		for j := range dc {
			temp_zip += int16(dc[j]) * int16(weight_matrix[i][j])
		}
		result[i][0] = temp_zip
	}
	return (result)
}


func internal_loop(dists []uint32, v uint32, k uint32,istart uint32,iend uint32, wg *sync.WaitGroup) {
	go func(){
		(*wg).Done()

	}()
}

//export makeresultmatrix
func makeresultmatrix(r int, c int) [][]int16 {
	a := make([][]int16, r)
	for i := range a {
		a[i] = make([]int16, c)
	}
	return a
}

func init_array(arr [][]int8) {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	for i:= range arr {
		for j:= range arr[i] {
			arr[i][j] = int8(r1.Intn(127))
		}
	}
}
func makeTimestamp() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

func makeMatrix(r int, c int) [][]int8 {
	a :=make([][]int8, r)
	for i:=range a {
		a[i] = make ([]int8,c)

	}
	init_array(a)
	return a
}
//export try_sample
func try_sample(){
	weight := makeMatrix(512,784);
	init_array(weight)
	data := makeMatrix(784, 1)
	init_array(data)
	result := multiply_matrix(weight, data)
	fmt.Print(result)

}

func main() {
	weight := makeMatrix(512,784);
	init_array(weight)
	data := makeMatrix(784, 1)
	init_array(data)
	multiply_matrix(weight, data)
}
