from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_circuit_layout, plot_bloch_multivector, plot_histogram
from qiskit_aer import Aer
from qiskit.visualization import plot_state_city  # 使用する可視化ツールを選択

# OpenQASMファイルから量子回路を読み込む
circuit = QuantumCircuit.from_qasm_file('example.qasm')

# 実行するバックエンドを指定
backend = Aer.get_backend('qasm_simulator')

# 回路をトランスパイルして、バックエンドに合わせたレイアウトを取得
transpiled_circuit = transpile(circuit, backend)

# 量子回路のレイアウトを表示
plot_circuit_layout(transpiled_circuit, backend)

# 実行した場合の測定結果を表示
job = backend.run(transpiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts(transpiled_circuit)
plot_histogram(counts)

# ブロッホ球の状態を表示
# 状態ベクトルを取得して表示するためには、理論的な状態ベクトルの計算やシミュレーションが必要
# ここでは、別の方法で状態ベクトルを取得する例を示します。
from qiskit import assemble
from qiskit.visualization import plot_state_hinton

# 回路を状態ベクトルシミュレータで実行
statevector_simulator = Aer.get_backend('statevector_simulator')
statevector_job = backend.run(transpile(circuit, statevector_simulator))
statevector = statevector_job.result().get_statevector()

# 状態ベクトルを表示
plot_bloch_multivector(statevector)
