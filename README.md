# my_quantum_simulator_wiθ_gpu
investigation

## Requirements
do this

### yaml
please install and build yaml.
```zsh
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build
cd build
cmake ..
make
sudo make install

```

### OpenMP
```zsh
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

### others
```zsh
pip install -r requirements.txt
```



