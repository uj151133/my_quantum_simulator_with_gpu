package main

import "C"
import "runtime"

//export Yield
func Yield() {
	runtime.Gosched()
}
